from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py

# -------------------------
# Units / constants
# -------------------------
FT_PER_M = 3.28084

# CS-LSTM grid constants (feet)
GRID_X, GRID_Y = 13, 3
R_LONG_FT = 90.0
BIN_FT = 15.0

# 논문과 동일 step-rate(5Hz)에서의 maneuver window (초)
LAT_LOOK_SEC = 4.0
LON_HIST_SEC = 3.0
LON_FUT_SEC  = 5.0

def get_neighbor_features(ego_hist, nbr_hist, grid_idx, fps):
    """
    nbr_hist: (T, 7) -> [frame, x, y, vx, vy, ax, ay]
    Returns: (T, 9) -> [rel_x, rel_y, rel_vx, abs_vy, ax, ay, lc, dxt, gate]
    """
    # 1. Basic Kinematics (Ego 기준 상대 정보)
    rel_x = nbr_hist[:, 1] - ego_hist[:, 1]
    rel_y = nbr_hist[:, 2] - ego_hist[:, 2]
    rel_vx = nbr_hist[:, 3] - ego_hist[:, 3] 
    abs_vy = nbr_hist[:, 4]                  
    ax = nbr_hist[:, 5]
    ay = nbr_hist[:, 6]

    # 2. Constants
    VY_EPS = 0.27 * 3.28084 # 0.27 m/s -> ft/s
    T_FRONT = 3.0
    T_BACK = 5.0
    EPS_GATE = 0.1 

    # 3. LC State
    lc_state = np.zeros_like(abs_vy)
    if grid_idx < 13: # Left Lane
        lc_state[abs_vy > VY_EPS] = -1.0  
        lc_state[abs_vy < -VY_EPS] = -3.0 
        lc_state[(abs_vy >= -VY_EPS) & (abs_vy <= VY_EPS)] = -2.0
    elif grid_idx >= 26: # Right Lane
        lc_state[abs_vy < -VY_EPS] = 1.0  
        lc_state[abs_vy > VY_EPS] = 3.0   
        lc_state[(abs_vy >= -VY_EPS) & (abs_vy <= VY_EPS)] = 2.0
    else: # Center
        lc_state[:] = 0.0

    # 4. dx_time & Gate
    denom = rel_vx.copy()
    denom[denom >= 0] += EPS_GATE
    denom[denom < 0] -= EPS_GATE
    dx_time = rel_x / denom
    
    gate = np.zeros_like(dx_time)
    gate[(-T_BACK < dx_time) & (dx_time < T_FRONT)] = 1.0

    return np.stack([rel_x, rel_y, rel_vx, abs_vy, ax, ay, lc_state, dx_time, gate], axis=1)

# -------------------------
# Maneuver labels
# -------------------------
def compute_lat_maneuver(lane_seq: np.ndarray, idx: int, fps: float) -> int:
    w = int(round(LAT_LOOK_SEC * fps))
    lb = max(0, idx - w)
    ub = min(len(lane_seq) - 1, idx + w)
    if lane_seq[ub] > lane_seq[idx] or lane_seq[idx] > lane_seq[lb]:
        return 3 # Right Change
    if lane_seq[ub] < lane_seq[idx] or lane_seq[idx] < lane_seq[lb]:
        return 2 # Left Change
    return 1 # Keep

def compute_lon_maneuver(s_seq: np.ndarray, idx: int, fps: float) -> int:
    lb = max(0, idx - int(round(LON_HIST_SEC * fps)))
    ub = min(len(s_seq) - 1, idx + int(round(LON_FUT_SEC  * fps)))
    if lb == idx or ub == idx:
        return 1
    v_hist = (s_seq[idx] - s_seq[lb]) / max(1, (idx - lb))
    v_fut  = (s_seq[ub]  - s_seq[idx]) / max(1, (ub - idx))
    if v_hist != 0 and (v_fut / v_hist) < 0.8:
        return 2 # Brake
    return 1 # Normal

# -------------------------
# DataFrame Preprocessing
# -------------------------
def apply_upper_flip_like_npz(df: pd.DataFrame) -> pd.DataFrame:
    upper = df["drivingDirection"].fillna(2).astype(int) == 1
    if not upper.any():
        return df
    x_max = float(df["x_ft"].max())
    c_y = 2.0 * float(df.loc[upper, "y_ft"].mean())
    upper_min_lane = int(df.loc[upper, "laneId"].min())
    upper_max_lane = int(df.loc[upper, "laneId"].max())

    df.loc[upper, "x_ft"] = x_max - df.loc[upper, "x_ft"].to_numpy()
    df.loc[upper, "y_ft"] = c_y  - df.loc[upper, "y_ft"].to_numpy()
    df.loc[upper, "laneId"] = (upper_min_lane + upper_max_lane) - df.loc[upper, "laneId"].astype(int)
    # Velocity/Accel flip
    df.loc[upper, "xVelocity"] = -df.loc[upper, "xVelocity"]
    df.loc[upper, "yVelocity"] = -df.loc[upper, "yVelocity"]
    df.loc[upper, "xAcceleration"] = -df.loc[upper, "xAcceleration"]
    df.loc[upper, "yAcceleration"] = -df.loc[upper, "yAcceleration"]
    return df

def downsample_frames(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    frame_min = int(df["frame"].min())
    keep = ((df["frame"] - frame_min) % stride) == 0
    return df.loc[keep].copy()

def build_social_grid_ids(frame_df: pd.DataFrame, ego: pd.Series) -> np.ndarray:
    grid = np.zeros(GRID_X * GRID_Y, dtype=np.float32)
    ego_id = int(ego["id"])
    ego_lane = int(ego["laneId"])
    ego_s = float(ego["x_ft"])
    lanes = [ego_lane - 1, ego_lane, ego_lane + 1]
    centers = np.array([-R_LONG_FT + i * BIN_FT for i in range(GRID_X)], dtype=np.float32)
    best_id = np.zeros(GRID_X * GRID_Y, dtype=np.int32)
    best_d  = np.full(GRID_X * GRID_Y, 1e18, dtype=np.float64)

    for _, nb in frame_df.iterrows():
        nb_id = int(nb["id"])
        if nb_id == ego_id: continue
        nb_lane = int(nb["laneId"])
        if nb_lane not in lanes: continue
        ds = float(nb["x_ft"]) - ego_s
        if abs(ds) > R_LONG_FT: continue
        lane_col = lanes.index(nb_lane)
        bin_idx = int(np.argmin(np.abs(centers - ds)))
        cell = lane_col * GRID_X + bin_idx
        d = float(abs(centers[bin_idx] - ds))
        if d < best_d[cell]:
            best_d[cell] = d
            best_id[cell] = nb_id
    grid[:] = best_id.astype(np.float32)
    return grid

def load_recording(tracks_csv: Path, tracks_meta_csv: Path, recording_meta_csv: Path,
                   ds_stride: int, target_fps: float) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(tracks_csv)
    tmeta = pd.read_csv(tracks_meta_csv)
    rmeta = pd.read_csv(recording_meta_csv)
    raw_fps = float(rmeta.loc[0, "frameRate"])

    if "drivingDirection" not in tmeta.columns:
        raise ValueError(f"{tracks_meta_csv} missing drivingDirection")
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id", how="left")

    df["xCenter_m"] = df["x"] + df["width"]  / 2.0
    df["yCenter_m"] = df["y"] + df["height"] / 2.0
    
    # Unit Conversion
    df["x_ft"] = df["xCenter_m"] * FT_PER_M
    df["y_ft"] = df["yCenter_m"] * FT_PER_M
    df["xVelocity"] = df["xVelocity"] * FT_PER_M
    df["yVelocity"] = df["yVelocity"] * FT_PER_M
    df["xAcceleration"] = df["xAcceleration"] * FT_PER_M
    df["yAcceleration"] = df["yAcceleration"] * FT_PER_M

    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["laneId"] = df["laneId"].astype(int)
    df["drivingDirection"] = df["drivingDirection"].astype(int)

    df = apply_upper_flip_like_npz(df)

    if abs(raw_fps - (target_fps * ds_stride)) > 1e-6:
        raise ValueError(f"Unexpected fps: raw_fps={raw_fps}, target_fps={target_fps}, stride={ds_stride}")
    df = downsample_frames(df, ds_stride)
    return df, target_fps

# -------------------------
# Main Building Logic (Tensor)
# -------------------------
def build_tensor_dataset(df: pd.DataFrame, fps: float, ds_id: int, t_h: int, t_f: int, slide_step: int):
    # laneId를 cols에 추가!
    cols = ["frame", "x_ft", "y_ft", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration", "laneId"]
    by_id = {}
    for vid, g in df.groupby("id"):
        # laneId 포함해서 저장
        by_id[int(vid)] = g[cols].sort_values("frame").to_numpy(dtype=np.float32) 

    by_frame = {int(fr): g for fr, g in df.groupby("frame")}

    out_ids, out_hist, out_fut, out_nbrs = [], [], [], []
    out_lat, out_lon, out_op_mask = [], [], []

    GRID_FLAT_SIZE = 39 

    for vid, track_data in by_id.items():
        total_len = len(track_data)
        frames = track_data[:, 0]
        lane_seq = track_data[:, 7] # laneId at index 7
        s_seq = track_data[:, 1]    # x_ft at index 1

        start_idx = t_h
        end_idx = total_len - t_f
        
        for i in range(start_idx, end_idx, slide_step):
            curr_fr = frames[i]
            ego_current_row = track_data[i]
            
            # 1. Ego Slice
            hist_idxs = slice(i - t_h, i + 1)
            fut_idxs = slice(i + 1, i + 1 + t_f)
            
            ego_hist_full = track_data[hist_idxs]
            ego_fut_full = track_data[fut_idxs]
            
            if len(ego_hist_full) != t_h + 1 or len(ego_fut_full) != t_f:
                continue

            # Ego Relative Pos (x, y at index 1, 2)
            ref_pos = ego_current_row[1:3] 
            ego_hist_xy = ego_hist_full[:, 1:3] - ref_pos
            ego_fut_xy = ego_fut_full[:, 1:3] - ref_pos

            # 2. Neighbors
            ego_series = df[(df.frame == curr_fr) & (df.id == vid)].iloc[0]
            grid_ids = build_social_grid_ids(by_frame[curr_fr], ego_series)

            nbr_tensor_sample = np.zeros((GRID_FLAT_SIZE, t_h + 1, 9), dtype=np.float32)

            for grid_k, nbr_id in enumerate(grid_ids):
                if nbr_id == 0 or nbr_id not in by_id: continue
                
                nbr_full_track = by_id[int(nbr_id)]
                ego_hist_frames = ego_hist_full[:, 0]
                
                mask = np.isin(nbr_full_track[:, 0], ego_hist_frames)
                nbr_segment = nbr_full_track[mask]
                
                if len(nbr_segment) == (t_h + 1):
                    # Full overlap: calculate features
                    feats = get_neighbor_features(ego_hist_full, nbr_segment, grid_k, fps)
                    nbr_tensor_sample[grid_k, :, :] = feats
                else:
                    # Partial overlap: fill matching parts
                    common_frames, ego_ind, nbr_ind = np.intersect1d(ego_hist_frames, nbr_full_track[:,0], return_indices=True)
                    if len(common_frames) > 0:
                        feats = get_neighbor_features(ego_hist_full[ego_ind], nbr_segment, grid_k, fps)
                        nbr_tensor_sample[grid_k, ego_ind, :] = feats

            # 3. Maneuvers
            lat_m = compute_lat_maneuver(lane_seq, i, fps)
            lon_m = compute_lon_maneuver(s_seq, i, fps)

            out_ids.append([ds_id, vid]) 
            out_hist.append(ego_hist_xy)
            out_fut.append(ego_fut_xy)
            out_nbrs.append(nbr_tensor_sample)
            out_lat.append(lat_m)
            out_lon.append(lon_m)
            out_op_mask.append(np.ones(t_f))

    # 빈 데이터 처리
    if not out_ids:
        return None

    return {
        "ids": np.array(out_ids, dtype=np.float32),
        "hist": np.stack(out_hist).astype(np.float32),
        "fut": np.stack(out_fut).astype(np.float32),
        "nbrs": np.stack(out_nbrs).astype(np.float32),
        "lat_enc": np.array(out_lat, dtype=np.float32), 
        "lon_enc": np.array(out_lon, dtype=np.float32),
        "op_mask": np.stack(out_op_mask).astype(np.float32)
    }

# -------------------------
# Split / Merge Helpers
# -------------------------
def merge_datasets(data_list: list[dict]) -> dict:
    if not data_list: return {}
    keys = data_list[0].keys()
    merged = {}
    for k in keys:
        merged[k] = np.concatenate([d[k] for d in data_list], axis=0)
    return merged

def slice_dataset(full_data: dict, ds_ids: list[int]) -> dict:
    if not full_data: return {}
    # dsId is in column 0 of 'ids'
    mask = np.isin(full_data['ids'][:, 0], ds_ids)
    sliced = {}
    for k, v in full_data.items():
        sliced[k] = v[mask]
    return sliced

def balanced_recording_split(ds_ids: list[int], ds_counts: dict[int,int], ratios=(0.8,0.1,0.1), seed=42):
    rng = np.random.default_rng(seed)
    total = sum(ds_counts[i] for i in ds_ids)
    targets = [total * r for r in ratios]
    items = [(i, ds_counts[i]) for i in ds_ids]
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)
    splits = {"train": [], "val": [], "test": []}
    sums = {"train": 0, "val": 0, "test": 0}
    keys = ["train", "val", "test"]
    for ds, cnt in items:
        deficits = {k: (targets[j] - sums[k]) for j, k in enumerate(keys)}
        best = max(deficits.items(), key=lambda kv: kv[1])[0]
        splits[best].append(ds)
        sums[best] += cnt
    return splits, sums, targets

def save_h5(out_path: Path, data_dict: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path.with_suffix('.h5'), 'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v, compression="gzip")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--highd_root", type=str, default="highD/raw")
    ap.add_argument("--out_root", type=str, default="highD/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_fps", type=float, default=5.0)
    ap.add_argument("--slide_window_sec", type=float, default=1.0)
    ap.add_argument("--smoke_n", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.highd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tracks_files = sorted(root.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(f"No *_tracks.csv found in {root}")

    # Accumulators
    processed_parts = [] # List of Dicts
    rec_names = []
    ds_id = 0
    ds_counts = {}

    for tf in tracks_files:
        rec = tf.name.split("_")[0]
        tmeta = root / f"{rec}_tracksMeta.csv"
        rmeta = root / f"{rec}_recordingMeta.csv"
        if not tmeta.exists() or not rmeta.exists(): continue

        raw_fps_temp = 25.0
        ds_stride = int(round(raw_fps_temp / args.target_fps))

        ds_id += 1
        df, eff_fps = load_recording(tf, tmeta, rmeta, ds_stride=ds_stride, target_fps=args.target_fps)

        # Build Tensors (Completed Logic)
        t_h = int(round(3.0 * eff_fps)) 
        t_f = int(round(5.0 * eff_fps)) 

        slide_step = int(round(args.slide_window_sec * eff_fps))
        
        data_part = build_tensor_dataset(df, eff_fps, ds_id, t_h, t_f, slide_step)
        
        if data_part is None:
            print(f"[SKIP] No samples in rec={rec}")
            continue

        processed_parts.append(data_part)
        rec_names.append(rec)
        count = len(data_part['ids'])
        ds_counts[ds_id] = count
        
        print(f"[OK] rec={rec} dsId={ds_id} rows={count:,}")

        # Smoke Test Saving
        if args.smoke_n and ds_id <= args.smoke_n:
            dbg_dir = out_root / "_smoke"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            sio.savemat(dbg_dir / f"TrainSet_{ds_id:02d}.mat", data_part)
            print(f"[SMOKE] Saved {dbg_dir / f'TrainSet_{ds_id:02d}.mat'}")
            if ds_id == args.smoke_n:
                print("[SMOKE] Done.")
                return

    if not processed_parts:
        raise RuntimeError("No data processed.")

    # Merge All
    print("Merging all recordings...")
    full_data = merge_datasets(processed_parts)
    ds_ids = list(ds_counts.keys())

    # Split 811
    #out_811 = out_root / "split_811"
    #out_811.mkdir(parents=True, exist_ok=True)
    #splits, sums, _ = balanced_recording_split(ds_ids, ds_counts, ratios=(0.8,0.1,0.1), seed=args.seed)
    
    # Save 811
    #print(f"Saving split_811... {sums}")
    #sio.savemat(out_811/"TrainSet.mat", slice_dataset(full_data, splits["train"]))
    #sio.savemat(out_811/"ValSet.mat",   slice_dataset(full_data, splits["val"]))
    #sio.savemat(out_811/"TestSet.mat",  slice_dataset(full_data, splits["test"]))

    # Split 712
    out_712 = out_root / "split_712"
    out_712.mkdir(parents=True, exist_ok=True)
    splits2, sums2, _ = balanced_recording_split(ds_ids, ds_counts, ratios=(0.7,0.1,0.2), seed=args.seed)

    # Save 712
    print(f"Saving split_712... {sums2}")
    save_h5(out_712/"TrainSet.mat", slice_dataset(full_data, splits2["train"]))
    save_h5(out_712/"ValSet.mat",   slice_dataset(full_data, splits2["val"]))
    save_h5(out_712/"TestSet.mat",  slice_dataset(full_data, splits2["test"]))

    print("[DONE]")

if __name__ == "__main__":
    main()