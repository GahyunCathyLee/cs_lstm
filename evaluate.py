from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CS-LSTM model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--measure_time', action='store_true', help='Measure inference time (batch_size=1, 1000 iters)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config, args.measure_time

def main():
    config, measure_time_mode = parse_args()
    args = config['model_args']
    args['train_flag'] = False

    paths  = config['data_paths']
    ds_args = config.get('dataset_args', {})
    nbr_mode = ds_args.get('nbr_mode', 0)

    mode_dim_map = {
        0: 2,   # (dx, dy)
        1: 5,   # (dax, day, lc_state, dx_time, gate)
        2: 7,   # (dx, dy, dvx, dvy, dax, day, gate)
        3: 2,   # (lc_state, dx_time)
        4: 13,  # all 13 dims
        5: 6,   # (dx, dy, dvx, dvy, dax, day)
        6: 7,   # (dx, dy, dvx, dvy, dax, day, I)
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
    # Support two path layouts (same as train.py):
    #   (A) mmap_dir + split_dir  — shared mmap + split indices
    #   (B) test_set              — legacy per-split mmap directory
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
    out_length = args['out_length']  # 15

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

            for b in range(fut.shape[0]):
                valid_idx = torch.where(mask_flat[b] > 0)[0]
                if len(valid_idx) > 0:
                    ade_total    += error[b, valid_idx].mean().item()
                    fde_total    += error[b, valid_idx[-1]].item()
                    total_samples += 1

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

if __name__ == '__main__':
    main()