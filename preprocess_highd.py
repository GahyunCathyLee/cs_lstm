from __future__ import annotations
import argparse
import bisect
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

# ==============================================================================
# Slot-based importance (ported from neighformer preprocess.py)
# ==============================================================================

# Grid-index → 8-slot ki mapping for slot weight lookup
# lane_group 0 (grid 0-12)  = left lane  → slots 2(preceding),3(alongside),4(following)
# lane_group 1 (grid 13-25) = same lane  → slots 0(preceding),1(following)
# lane_group 2 (grid 26-38) = right lane → slots 5(preceding),6(alongside),7(following)
def _grid_idx_to_slot_ki(grid_idx):
    lane_group  = grid_idx // GRID_X  # 0=left, 1=center, 2=right
    bin_in_lane = grid_idx % GRID_X   # 0=far_behind .. 6=alongside .. 12=far_ahead
    if lane_group == 1:   # same lane
        return 0 if bin_in_lane >= 7 else 1
    elif lane_group == 0: # left lane
        if bin_in_lane >= 7:   return 2   # leftPreceding
        elif bin_in_lane == 6: return 3   # leftAlongside
        else:                  return 4   # leftFollowing
    else:                 # right lane
        if bin_in_lane >= 7:   return 5   # rightPreceding
        elif bin_in_lane == 6: return 6   # rightAlongside
        else:                  return 7   # rightFollowing

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot, from dataset analysis)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

# Conditional slot weights derived from SlotWeightProbe models (mean softmax per slot).
# Used when --slot_importance_conditional is set.

# No-LC case: weights by ego lane level  (0=leftmost/fast, 1=middle, 2=rightmost/slow)
SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4657, 0.0163, 0.0000, 0.0000, 0.0000, 0.4357, 0.0035, 0.0788],  # ll0 leftmost
    [0.4240, 0.0346, 0.3347, 0.0197, 0.1859, 0.0007, 0.0002, 0.0001],  # ll1 middle
    [0.3846, 0.0141, 0.3593, 0.0345, 0.2070, 0.0000, 0.0000, 0.0000],  # ll2 rightmost
]

# LC-in-history case: pre-LC weights per lc_type (0-5)
SLOT_WEIGHTS_PRE_LC = [
    [0.0000, 0.0037, 0.0000, 0.0000, 0.0000, 0.2718, 0.1157, 0.6089],  # lct0 leftmost→middle
    [0.7023, 0.1658, 0.0000, 0.0000, 0.0000, 0.1251, 0.0049, 0.0019],  # lct1 leftmost→rightmost
    [0.3170, 0.0117, 0.0033, 0.0003, 0.0005, 0.5215, 0.0168, 0.1289],  # lct2 middle→leftmost
    [0.0367, 0.0057, 0.4062, 0.1076, 0.4435, 0.0000, 0.0000, 0.0001],  # lct3 middle→rightmost
    [0.9996, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # lct4 rightmost→leftmost
    [0.0048, 0.0000, 0.5762, 0.1229, 0.2962, 0.0000, 0.0000, 0.0000],  # lct5 rightmost→middle
]

# LC-in-history case: post-LC weights per lc_type (0-5)
SLOT_WEIGHTS_POST_LC = [
    [0.0017, 0.0074, 0.0026, 0.0011, 0.0109, 0.4849, 0.0611, 0.4303],  # lct0 leftmost→middle
    [0.0478, 0.0078, 0.7227, 0.0393, 0.1825, 0.0000, 0.0000, 0.0000],  # lct1 leftmost→rightmost
    [0.8647, 0.0680, 0.0000, 0.0000, 0.0000, 0.0527, 0.0042, 0.0103],  # lct2 middle→leftmost
    [0.0557, 0.9204, 0.0001, 0.0001, 0.0237, 0.0000, 0.0000, 0.0000],  # lct3 middle→rightmost
    [0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.9557, 0.0427, 0.0013],  # lct4 rightmost→leftmost
    [0.0125, 0.0334, 0.0001, 0.0016, 0.0006, 0.2424, 0.0296, 0.6799],  # lct5 rightmost→middle
]

# (from_level, to_level) → lc_type
_LC_TYPE_MAP_LEVEL = {
    (0, 1): 0, (0, 2): 1,
    (1, 0): 2, (1, 2): 3,
    (2, 0): 4, (2, 1): 5,
}

# LIS binning
LIS_BINS = {
    '3': {'cuts': [-5.8639, 4.9525],
          'vals': [-1.0, 0.0, 1.0]},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2.0, -1.0, 0.0, 1.0, 2.0]},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]},
}

IMPORTANCE_PARAMS_LIS = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}

IMPORTANCE_PARAMS_LIT = {
    'sx': 15.0, 'ax': 0.2, 'bx': 0.25,
    'sy':  2.0, 'ay': 0.01, 'by': 0.1,
}

_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]  # 4 inner cuts → 5 bins (0~4)

# Neighbor feature dimension: [dx,dy,dvx,abs_vy,ax,ay,lc_state,volume,size_bin,gate,I_x,I_y,I]
NB_DIM = 13


def _lit_to_lis(lit, lis_mode):
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


def _volume_bin(phys_length, phys_width, vehicle_class):
    """Return (size_bin 0~4, raw volume m³) for a vehicle."""
    if vehicle_class == "Car":
        if phys_length < 4.5:   height = 1.45
        elif phys_length < 5.0: height = 1.70
        else:                   height = 1.90
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate(_VOLUME_BIN_EDGES):
        if volume < edge:
            return float(i), volume
    return 4.0, volume


def compute_importance_lis(lis, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIS
    ix = (np.exp(-(lis ** 2) / (2.0 * p['sx'] ** 2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state ** 2) / (2.0 * p['sy'] ** 2))
          * np.exp(-p['ay'] * (abs(lis) ** p['py']))
          * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))


def compute_importance_lit(lit, delta_lane, lc_state):
    p = IMPORTANCE_PARAMS_LIT
    ix = (np.exp(-(lit ** 2) / (2.0 * p['sx'] ** 2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state ** 2) / (2.0 * p['sy'] ** 2))
          * np.exp(-p['ay'] * (abs(lit) ** 1.5))
          * np.exp(-p['by'] * delta_lane))
    return float(ix), float(iy), float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))


def _lane_id_to_level(lid, dd, sorted_lids, post_flip):
    """lane_id → lane_level (0=leftmost/fast, 1=middle, 2=rightmost/slow)."""
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1:
        return 1
    if post_flip or dd == 2:
        if idx == 0:     return 0
        if idx == n - 1: return 2
        return 1
    else:  # dd=1, no flip
        if idx == 0:     return 2
        if idx == n - 1: return 0
        return 1


def _ego_lc_context(ego_lane_arr, dd, lane_ids_per_dd, post_flip):
    """history window 내 ego LC 상태를 판단한다.

    Returns (lane_level, lc_frame_ti, lc_type)
      lane_level  : 0/1/2 (no-LC, ego의 t0 차선), -2 (LC in history), -1 (unknown)
      lc_frame_ti : LC가 처음 일어난 hist frame 인덱스 (None = no LC)
      lc_type     : 0-5  (-1 = no LC or unknown)
    """
    sorted_lids = lane_ids_per_dd.get(dd, [])
    lc_frame_ti = None
    lc_type = -1
    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), dd, sorted_lids, post_flip)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     dd, sorted_lids, post_flip)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break
    if lc_frame_ti is None:
        lane_level = _lane_id_to_level(int(ego_lane_arr[-1]), dd, sorted_lids, post_flip)
    else:
        lane_level = -2
    return lane_level, lc_frame_ti, lc_type


def _get_slot_weight(ki, ti, lane_level, lc_frame_ti, lc_type):
    """slot ki / timestep ti에 대응하는 조건부 slot weight를 반환."""
    if lc_frame_ti is not None and lc_type >= 0:
        if ti < lc_frame_ti:
            return SLOT_WEIGHTS_PRE_LC[lc_type][ki]
        else:
            return SLOT_WEIGHTS_POST_LC[lc_type][ki]
    elif 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    else:
        return SLOT_WEIGHTS[ki]  # fallback


def get_neighbor_features(ego_hist, nbr_hist, grid_idx, fps,
                           nb_len_ft, ego_len_ft, nb_volume, nb_size_bin, delta_lane,
                           args, _lc_lane_lv=-1, _lc_frame_ti=None, _lc_type=-1):
    """
    nbr_hist : (T, 8) → [frame, x_ft, y_ft, vx, vy, ax, ay, laneId]
    ego_hist : (T, 8) → same

    Returns  : (T, 13) → [rel_x, rel_y, rel_vx, abs_vy, ax, ay,
                           lc_state, volume, size_bin, gate, I_x, I_y, I]

    lc_state uses {0: closing in, 1: staying, 2: moving out} convention.
    Importance (I_x, I_y, I) computed via LIT (bumper-to-bumper, ft/s units)
    converted to LIS, using the same formula as neighformer/other models.
    """
    T = len(ego_hist)

    # ── Basic kinematics ─────────────────────────────────────────────────────
    rel_x  = nbr_hist[:, 1] - ego_hist[:, 1]
    rel_y  = nbr_hist[:, 2] - ego_hist[:, 2]
    rel_vx = nbr_hist[:, 3] - ego_hist[:, 3]
    abs_vy = nbr_hist[:, 4]
    ax     = nbr_hist[:, 5]
    ay     = nbr_hist[:, 6]

    VY_EPS  = args.vy_eps * FT_PER_M   # m/s → ft/s
    EPS_LIT = args.eps_gate * FT_PER_M  # m/s → ft/s (LIT denominator clamp)

    # ── lc_state: {0: closing in, 1: staying, 2: moving out} ─────────────────
    lc_state = np.ones(T, dtype=np.float32)  # default: staying
    if grid_idx < GRID_X:            # left lane  (ego lane - 1)
        lc_state[abs_vy >  VY_EPS] = 0.0  # moving toward ego lane
        lc_state[abs_vy < -VY_EPS] = 2.0  # moving away
    elif grid_idx >= 2 * GRID_X:     # right lane (ego lane + 1)
        lc_state[abs_vy < -VY_EPS] = 0.0  # moving toward ego lane
        lc_state[abs_vy >  VY_EPS] = 2.0  # moving away
    # else center: lc_state stays 1.0

    # ── Bumper-to-bumper LIT (seconds, unit-independent via ft/s) ────────────
    half_sum_ft = 0.5 * (ego_len_ft + nb_len_ft)
    gap_ft = np.where(rel_x >= 0,
                      np.abs(rel_x - half_sum_ft),
                      np.abs(-rel_x - half_sum_ft))
    denom_base = np.where(rel_x >= 0, rel_vx, -rel_vx)
    denom = np.where(denom_base >= 0,
                     denom_base + EPS_LIT,
                     denom_base - EPS_LIT)
    lit_arr = gap_ft / denom

    # ── Per-timestep importance ───────────────────────────────────────────────
    ki      = _grid_idx_to_slot_ki(grid_idx)
    I_x_arr  = np.zeros(T, dtype=np.float32)
    I_y_arr  = np.zeros(T, dtype=np.float32)
    I_arr    = np.zeros(T, dtype=np.float32)
    gate_arr = np.zeros(T, dtype=np.float32)

    for ti in range(T):
        lc_s = float(lc_state[ti])
        lit  = float(lit_arr[ti])
        lis  = _lit_to_lis(lit, args.lis_mode)

        if args.importance_mode == 'lit':
            ix, iy, i_total = compute_importance_lit(lit, delta_lane, lc_s)
        else:
            ix, iy, i_total = compute_importance_lis(lis, delta_lane, lc_s)

        # ── slot importance boost ─────────────────────────────────────────
        if args.slot_importance_alpha > 0.0:
            if args.slot_importance_conditional:
                w_slot = _get_slot_weight(ki, ti, _lc_lane_lv, _lc_frame_ti, _lc_type)
            else:
                w_slot = SLOT_WEIGHTS[ki]
            i_total = min(i_total * (1.0 + args.slot_importance_alpha * w_slot), 1.0)

        # ── gate ─────────────────────────────────────────────────────────
        g = 1.0 if (args.gate_theta <= 0.0 or i_total >= args.gate_theta) else 0.0

        I_x_arr[ti]  = ix * g
        I_y_arr[ti]  = iy * g
        I_arr[ti]    = i_total * g
        gate_arr[ti] = g

    vol_arr  = np.full(T, nb_volume,  dtype=np.float32)
    sbin_arr = np.full(T, nb_size_bin, dtype=np.float32)

    return np.stack([rel_x, rel_y, rel_vx, abs_vy, ax, ay,
                     lc_state, vol_arr, sbin_arr, gate_arr,
                     I_x_arr, I_y_arr, I_arr], axis=1)


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

    # Rename tmeta physical dimensions to avoid conflict with tracks bbox columns
    tmeta = tmeta.rename(columns={"width": "phys_length_m", "height": "phys_width_m"})

    merge_cols = ["id", "drivingDirection", "phys_length_m", "phys_width_m"]
    if "class" in tmeta.columns:
        merge_cols.append("class")
    df = df.merge(tmeta[merge_cols], on="id", how="left")

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
def build_tensor_dataset(df: pd.DataFrame, fps: float, ds_id: int, t_h: int, t_f: int,
                         slide_step: int, args):
    # Trajectory columns: frame, x_ft, y_ft, vx, vy, ax, ay, laneId, drivingDirection
    cols = ["frame", "x_ft", "y_ft", "xVelocity", "yVelocity",
            "xAcceleration", "yAcceleration", "laneId", "drivingDirection"]
    by_id = {}
    for vid, g in df.groupby("id"):
        by_id[int(vid)] = g[cols].sort_values("frame").to_numpy(dtype=np.float32)

    # Vehicle meta: physical length (ft), volume (m³), size_bin
    by_id_meta = {}  # {vid: (len_ft, volume, size_bin)}
    for vid, g in df.groupby("id"):
        row = g.iloc[0]
        len_m = float(row.get("phys_length_m", 0.0))
        wid_m = float(row.get("phys_width_m", 0.0))
        cls   = str(row.get("class", "Car")) if "class" in df.columns else "Car"
        size_bin, volume = _volume_bin(len_m, wid_m, cls)
        by_id_meta[int(vid)] = (len_m * FT_PER_M, volume, size_bin)

    # drivingDirection per vehicle (for ego_lc_context)
    vid_to_dd = {}
    for vid, g in df.groupby("id"):
        vid_to_dd[int(vid)] = int(g["drivingDirection"].iloc[0])

    by_frame = {int(fr): g for fr, g in df.groupby("frame")}

    # per-dd sorted lane IDs (for conditional slot weights)
    lane_ids_per_dd = {}
    if args.slot_importance_conditional:
        for dd_val in [1, 2]:
            lids = sorted(set(int(x) for x in df.loc[df["drivingDirection"] == dd_val, "laneId"] if x > 0))
            lane_ids_per_dd[dd_val] = lids

    out_ids, out_hist, out_fut, out_nbrs = [], [], [], []
    out_lat, out_lon, out_op_mask = [], [], []

    GRID_FLAT_SIZE = GRID_X * GRID_Y  # 39

    for vid, track_data in by_id.items():
        total_len = len(track_data)
        frames    = track_data[:, 0]
        lane_seq  = track_data[:, 7]  # laneId
        dd_seq    = track_data[:, 8]  # drivingDirection
        s_seq     = track_data[:, 1]  # x_ft

        ego_len_ft, ego_volume, ego_size_bin = by_id_meta.get(vid, (0.0, 0.0, 0.0))

        start_idx = t_h
        end_idx   = total_len - t_f

        for i in range(start_idx, end_idx, slide_step):
            curr_fr      = frames[i]
            ego_curr_row = track_data[i]

            # 1. Ego Slice
            hist_idxs = slice(i - t_h, i + 1)
            fut_idxs  = slice(i + 1, i + 1 + t_f)

            ego_hist_full = track_data[hist_idxs]
            ego_fut_full  = track_data[fut_idxs]

            if len(ego_hist_full) != t_h + 1 or len(ego_fut_full) != t_f:
                continue

            # Ego Relative Pos
            ref_pos      = ego_curr_row[1:3]
            ego_hist_xy  = ego_hist_full[:, 1:3] - ref_pos
            ego_fut_xy   = ego_fut_full[:, 1:3]  - ref_pos

            # ── conditional slot weight context ──────────────────────────────
            _lc_lane_lv, _lc_frame_ti, _lc_type = -1, None, -1
            if args.slot_importance_conditional and args.slot_importance_alpha > 0.0:
                ego_dd       = vid_to_dd.get(vid, 2)
                ego_lane_arr = ego_hist_full[:, 7].astype(np.int32)
                _lc_lane_lv, _lc_frame_ti, _lc_type = _ego_lc_context(
                    ego_lane_arr, ego_dd, lane_ids_per_dd, True  # normalize_flip=True
                )

            # 2. Neighbors
            ego_series = df[(df.frame == curr_fr) & (df.id == vid)].iloc[0]
            grid_ids   = build_social_grid_ids(by_frame[curr_fr], ego_series)

            nbr_tensor_sample = np.zeros((GRID_FLAT_SIZE, t_h + 1, NB_DIM), dtype=np.float32)

            for grid_k, nbr_id in enumerate(grid_ids):
                if nbr_id == 0 or nbr_id not in by_id:
                    continue

                nb_len_ft, nb_volume, nb_size_bin = by_id_meta.get(int(nbr_id), (0.0, 0.0, 0.0))
                # delta_lane: 0 for same lane (grid_k 13-25), 1 for adjacent lanes
                delta_lane = 0 if (GRID_X <= grid_k < 2 * GRID_X) else 1

                nbr_full_track  = by_id[int(nbr_id)]
                ego_hist_frames = ego_hist_full[:, 0]

                mask        = np.isin(nbr_full_track[:, 0], ego_hist_frames)
                nbr_segment = nbr_full_track[mask]

                if len(nbr_segment) == (t_h + 1):
                    feats = get_neighbor_features(
                        ego_hist_full, nbr_segment, grid_k, fps,
                        nb_len_ft, ego_len_ft, nb_volume, nb_size_bin, delta_lane,
                        args, _lc_lane_lv, _lc_frame_ti, _lc_type,
                    )
                    nbr_tensor_sample[grid_k, :, :] = feats
                else:
                    common_frames, ego_ind, nbr_ind = np.intersect1d(
                        ego_hist_frames, nbr_full_track[:, 0], return_indices=True
                    )
                    if len(common_frames) > 0:
                        feats = get_neighbor_features(
                            ego_hist_full[ego_ind], nbr_segment, grid_k, fps,
                            nb_len_ft, ego_len_ft, nb_volume, nb_size_bin, delta_lane,
                            args, _lc_lane_lv, _lc_frame_ti, _lc_type,
                        )
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

    if not out_ids:
        return None

    return {
        "ids":     np.array(out_ids, dtype=np.float32),
        "hist":    np.stack(out_hist).astype(np.float32),
        "fut":     np.stack(out_fut).astype(np.float32),
        "nbrs":    np.stack(out_nbrs).astype(np.float32),
        "lat_enc": np.array(out_lat, dtype=np.float32),
        "lon_enc": np.array(out_lon, dtype=np.float32),
        "op_mask": np.stack(out_op_mask).astype(np.float32),
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
    ap.add_argument("--highd_root",       type=str,   default="highD/raw")
    ap.add_argument("--out_root",         type=str,   default="highD/processed")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--target_fps",       type=float, default=5.0)
    ap.add_argument("--slide_window_sec", type=float, default=1.0)
    ap.add_argument("--smoke_n",          type=int,   default=0)

    # importance / gate (aligned with neighformer defaults)
    ap.add_argument("--vy_eps",        type=float, default=0.27,
                    help="lateral velocity threshold for lc_state (m/s, converted to ft/s internally)")
    ap.add_argument("--eps_gate",      type=float, default=1.0,
                    help="eps for LIT denominator clamp (m/s, converted to ft/s internally)")
    ap.add_argument("--lis_mode",      type=str,   default="3", choices=["3", "5", "7", "9"],
                    help="LIS binning mode")
    ap.add_argument("--importance_mode", type=str, default="lis", choices=["lis", "lit"],
                    help="lis=use discrete LIS | lit=use continuous LIT (legacy params)")
    ap.add_argument("--gate_theta",    type=float, default=0.0,
                    help="importance threshold gate (0.0 = all active)")
    ap.add_argument("--slot_importance", type=float, default=0.0, dest="slot_importance_alpha",
                    help="slot importance boost alpha: I_new = min(I*(1+alpha*w_slot), 1.0). 0.0=disabled")
    ap.add_argument("--slot_importance_conditional", action="store_true", default=False,
                    help="use lane-level/pre-LC/post-LC conditional slot weights")

    args = ap.parse_args()

    root     = Path(args.highd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tracks_files = sorted(root.glob("*_tracks.csv"))
    if not tracks_files:
        raise FileNotFoundError(f"No *_tracks.csv found in {root}")

    print(f"[Config] importance_mode={args.importance_mode}  lis_mode={args.lis_mode}  "
          f"gate_theta={args.gate_theta}")
    print(f"         slot_alpha={args.slot_importance_alpha}"
          + ("  slot_conditional=True" if args.slot_importance_conditional else ""))
    print(f"         nbrs shape per sample: (39, T, {NB_DIM})")

    processed_parts = []
    rec_names = []
    ds_id = 0
    ds_counts = {}

    for tf in tracks_files:
        rec   = tf.name.split("_")[0]
        tmeta = root / f"{rec}_tracksMeta.csv"
        rmeta = root / f"{rec}_recordingMeta.csv"
        if not tmeta.exists() or not rmeta.exists(): continue

        raw_fps_temp = 25.0
        ds_stride    = int(round(raw_fps_temp / args.target_fps))

        ds_id += 1
        df, eff_fps = load_recording(tf, tmeta, rmeta, ds_stride=ds_stride, target_fps=args.target_fps)

        t_h        = int(round(3.0 * eff_fps))
        t_f        = int(round(5.0 * eff_fps))
        slide_step = int(round(args.slide_window_sec * eff_fps))

        data_part = build_tensor_dataset(df, eff_fps, ds_id, t_h, t_f, slide_step, args)

        if data_part is None:
            print(f"[SKIP] No samples in rec={rec}")
            continue

        processed_parts.append(data_part)
        rec_names.append(rec)
        count = len(data_part['ids'])
        ds_counts[ds_id] = count

        print(f"[OK] rec={rec} dsId={ds_id} rows={count:,}")

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

    print("Merging all recordings...")
    full_data = merge_datasets(processed_parts)
    ds_ids    = list(ds_counts.keys())

    # Split 712
    out_712 = out_root / "split_712"
    out_712.mkdir(parents=True, exist_ok=True)
    splits2, sums2, _ = balanced_recording_split(ds_ids, ds_counts, ratios=(0.7, 0.1, 0.2), seed=args.seed)

    print(f"Saving split_712... {sums2}")
    save_h5(out_712 / "TrainSet.mat", slice_dataset(full_data, splits2["train"]))
    save_h5(out_712 / "ValSet.mat",   slice_dataset(full_data, splits2["val"]))
    save_h5(out_712 / "TestSet.mat",  slice_dataset(full_data, splits2["test"]))

    print("[DONE]")

if __name__ == "__main__":
    main()
