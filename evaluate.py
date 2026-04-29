from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import yaml
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Scenario label helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_scenario_labels(path):
    """
    scenario_labels.csv → labels_lut dict.
    Returns None if the file is missing or lacks required columns.

    CSV 필수 컬럼: recordingId, trackId, t0_frame
    선택 컬럼   : event_label, state_label
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path} → stratified eval disabled")
        return None

    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing} → stratified eval disabled")
        return None

    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label → stratified eval disabled")
        return None

    lut = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }

    print(f"\n[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


def build_sample_label_list(mmap_dir, indices, labels_lut):
    """
    mmap_dir의 meta npy 파일과 split indices를 이용해
    각 샘플의 (event_label, state_label)을 순서대로 반환.

    Returns list of dicts (or None per entry if no match).
    """
    mmap_dir = Path(mmap_dir)
    meta_rec   = np.load(mmap_dir / "meta_recordingId.npy", mmap_mode='r')
    meta_track = np.load(mmap_dir / "meta_trackId.npy",     mmap_mode='r')
    meta_frame = np.load(mmap_dir / "meta_frame.npy",       mmap_mode='r')

    sample_labels = []
    for idx in indices:
        key = (int(meta_rec[idx]), int(meta_track[idx]), int(meta_frame[idx]))
        sample_labels.append(labels_lut.get(key))
    return sample_labels


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * w for w in widths) + right


def print_scenario_results(stats, label_type):
    """
    Print per-scenario ADE / FDE / RMSE table.

    stats      : {label: [sum_ade, sum_fde, sum_rmse, count]}
    label_type : "Event" or "State"
    """
    if not stats:
        return

    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))

    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2
    c_n   = 9
    c_m   = 11

    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(
        f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
        f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|"
    )
    print(_sep(ws))

    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(
            f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
            f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|"
        )

    print(_sep(ws))

    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    total_n  = sum(v[3] for v in stats.values())
    N = max(1, total_n)
    print(
        f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
        f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|"
    )
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Parse args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CS-LSTM model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--measure_time', action='store_true', help='Measure inference time (batch_size=1, 1000 iters)')
    parser.add_argument('--scenario_labels', type=str, default=None,
                        help='Path to scenario_labels.csv for per-scenario breakdown')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config, args.measure_time, args.scenario_labels


def main():
    config, measure_time_mode, scenario_labels_arg = parse_args()
    args = config['model_args']
    args['train_flag'] = False

    paths  = config['data_paths']
    ds_args = config.get('dataset_args', {})
    nbr_mode = ds_args.get('nbr_mode', 0)

    mode_dim_map = {
        0: 2,   # (dx, dy)
        1: 5,   # (dax, day, s_x, s_y, I)
        2: 7,   # (dx, dy, dvx, dvy, dax, day, gate)
        3: 2,   # (s_x, s_y)
        4: 10,  # all 10 dims
        5: 6,   # (dx, dy, dvx, dvy, dax, day)
        6: 7,   # (dx, dy, dvx, dvy, dax, day, I)
        7: 7,
        8: 8,   # (dx, dy, dvx, dvy, dax, day, dim, I)
    }
    current_nbr_dim = mode_dim_map.get(nbr_mode, 2)
    args['nbr_input_dim'] = current_nbr_dim
    print(f"✅ Mode: {nbr_mode} | Neighbor Input Dim: {current_nbr_dim}")

    if args.get('use_maneuvers', False):
        print("[WARN] 새 전처리 데이터에는 maneuver 레이블이 없습니다. "
              "use_maneuvers=False 로 강제합니다.")
        args['use_maneuvers'] = False

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    net = highwayNet(args)
    device = torch.device("cuda" if args['use_cuda'] and torch.cuda.is_available() else "cpu")
    ckpt_path = paths['save_dir'] + "/best.pt"
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net = net.to(device)
    net.eval()

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    mmap_dir  = paths.get('mmap_dir', None)
    split_dir = paths.get('split_dir', None)

    if mmap_dir is not None:
        ts_indices = None
        if split_dir is not None:
            ts_indices = np.load(Path(split_dir) / 'test_indices.npy')
        tsSet = ngsimDataset(mmap_dir,
                             enc_size=args['encoder_size'],
                             grid_size=tuple(args['grid_size']),
                             nbr_feature_mode=nbr_mode, indices=ts_indices)
    else:
        tsSet = ngsimDataset(paths['test_set'],
                             enc_size=args['encoder_size'],
                             grid_size=tuple(args['grid_size']),
                             nbr_feature_mode=nbr_mode)
        ts_indices = None
        mmap_dir   = None
    out_length = args['out_length']  # 15

    # ── 시나리오 레이블 로드 ──────────────────────────────────────────────
    labels_lut    = None
    sample_labels = None  # per-sample label list aligned with test set order
    scenario_labels_path = scenario_labels_arg or paths.get('scenario_labels', None)

    if scenario_labels_path and not measure_time_mode:
        labels_lut = load_scenario_labels(scenario_labels_path)
        if labels_lut is not None and mmap_dir is not None:
            sample_labels = build_sample_label_list(mmap_dir, tsSet.indices, labels_lut)

    # =====================================================================
    # Inference Time 측정 모드
    # =====================================================================
    if measure_time_mode:
        print("\n⏳ Measuring Inference Time (Batch Size = 1)")
        time_loader = DataLoader(tsSet, batch_size=1, shuffle=True,
                                 num_workers=4, collate_fn=tsSet.collate_fn)

        num_iterations  = 1000
        warmup_iterations = 100
        inference_times = []

        with torch.no_grad():
            for i, data in enumerate(time_loader):
                if i >= num_iterations + warmup_iterations:
                    break
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
                hist    = hist.to(device)
                nbrs    = nbrs.to(device)
                mask    = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)

                if args['use_cuda']:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                _ = net(hist, nbrs, mask, lat_enc, lon_enc)

                if args['use_cuda']:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                if i >= warmup_iterations:
                    inference_times.append((end_time - start_time) * 1000)

        avg_time = sum(inference_times) / len(inference_times)
        print(f"✅ Result ({num_iterations} iterations):")
        print(f"  - Average Inference Time: {avg_time:.2f} ms")
        print(f"  - Min / Max Time: {min(inference_times):.2f} ms / {max(inference_times):.2f} ms\n")
        return

    # =====================================================================
    # 일반 평가 모드
    # =====================================================================
    batch_size   = config['train_args']['batch_size']
    tsDataloader = DataLoader(tsSet, batch_size=batch_size, shuffle=False,
                              num_workers=8, collate_fn=tsSet.collate_fn)

    nllLoss = torch.zeros(out_length).to(device)
    mseLoss = torch.zeros(out_length).to(device)
    counts  = torch.zeros(out_length).to(device)

    ade_total    = 0.0
    fde_total    = 0.0
    total_samples = 0

    # [sum_ade, sum_fde, sum_rmse, count]
    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    sample_cursor = 0   # tracks position in sample_labels

    eval_loop = tqdm(tsDataloader, desc="Evaluating")

    for data in eval_loop:
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        # collate_fn 출력: (Tf, B, 2) → maskedNLLTest/maskedMSETest는 (B, Tf, 2) 기대
        fut     = fut.permute(1, 0, 2)      # (B, Tf, 2)
        op_mask = op_mask.permute(1, 0, 2)  # (B, Tf, 2)

        hist    = hist.to(device)
        nbrs    = nbrs.to(device)
        mask    = mask.to(device)
        lat_enc = lat_enc.to(device)
        lon_enc = lon_enc.to(device)
        fut     = fut.to(device)
        op_mask = op_mask.to(device)

        with torch.no_grad():
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            # fut_pred: (Tf, B, 5) → permute to (B, Tf, 5) for loss fns
            fut_pred = fut_pred.permute(1, 0, 2)

            l_nll, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,
                                     use_maneuvers=False)
            l_mse, _ = maskedMSETest(fut_pred, fut, op_mask)

            # ADE / FDE  (ft → m: × 0.3048)
            error    = torch.norm(fut_pred[:, :, :2] - fut, p=2, dim=2) * 0.3048
            mask_flat = op_mask[:, :, 0]  # (B, Tf)

            B = fut.shape[0]
            for b in range(B):
                valid_idx = torch.where(mask_flat[b] > 0)[0]
                if len(valid_idx) > 0:
                    b_ade = error[b, valid_idx].mean().item()
                    b_fde = error[b, valid_idx[-1]].item()
                    b_rmse = torch.sqrt((error[b, valid_idx] ** 2).mean()).item()

                    ade_total    += b_ade
                    fde_total    += b_fde
                    total_samples += 1

                    # ── 시나리오별 누적 ──────────────────────────────────
                    if sample_labels is not None:
                        lab = sample_labels[sample_cursor + b]
                        if lab is not None:
                            ev = lab.get("event_label") or "unknown"
                            st = lab.get("state_label") or "unknown"
                            ev_stats[ev][0] += b_ade
                            ev_stats[ev][1] += b_fde
                            ev_stats[ev][2] += b_rmse
                            ev_stats[ev][3] += 1
                            st_stats[st][0] += b_ade
                            st_stats[st][1] += b_fde
                            st_stats[st][2] += b_rmse
                            st_stats[st][3] += 1

        sample_cursor += B

        nllLoss += l_nll.detach()
        mseLoss += l_mse.detach()
        counts  += c.detach()

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    valid_mask = counts > 0

    avg_nll = (nllLoss[valid_mask] / counts[valid_mask]).mean().item()

    mse_per_frame  = mseLoss[valid_mask] / counts[valid_mask]
    rmse_per_frame = torch.pow(mse_per_frame, 0.5) * 0.3048  # ft → m
    avg_rmse = rmse_per_frame.mean().item()

    avg_ade = ade_total / total_samples if total_samples > 0 else 0
    avg_fde = fde_total / total_samples if total_samples > 0 else 0

    # target_hz=3 기준 1s 간격 인덱스: 2, 5, 8, 11, 14
    hz = ds_args.get('target_hz', 3)
    snap_indices = [int(round((s + 1) * hz)) - 1 for s in range(5)]  # 1s~5s

    print(f"\nFinal Evaluation Results:")
    print(f"  Average NLL          : {avg_nll:.4f}")
    print(f"  Average RMSE (m)     : {avg_rmse:.4f}")

    rmse_snaps = []
    for i, idx in enumerate(snap_indices):
        if idx < len(rmse_per_frame):
            rmse_snaps.append(f"@{i+1}s: {rmse_per_frame[idx].item():.4f}")
    print("  RMSE snapshots (m)   : " + ",  ".join(rmse_snaps))

    print(f"  Average ADE (m)      : {avg_ade:.4f}")
    print(f"  Average FDE (m)      : {avg_fde:.4f}")

    # ── 시나리오별 결과 출력 ──────────────────────────────────────────────
    if ev_stats:
        print_scenario_results(ev_stats, label_type="Event")
    if st_stats:
        print_scenario_results(st_stats, label_type="State")

if __name__ == '__main__':
    main()
