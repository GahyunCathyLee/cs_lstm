from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path

# -------------------------------------------------------------------------
# 13×3 소셜 그리드 상수 (model.py / 기존 CS-LSTM과 동일)
# -------------------------------------------------------------------------
GRID_X      = 13
GRID_Y      = 3
R_LONG      = 90.0   # ft  (longitudinal half-range)
BIN_FT      = 15.0   # ft  (bin width)
BIN_CENTERS = np.array([-R_LONG + i * BIN_FT for i in range(GRID_X)],
                        dtype=np.float32)   # [-90, -75, ..., 90]

# 새 전처리의 8 슬롯 → lane_row 매핑
# lane_row: 0=left, 1=center(same), 2=right  (기존 collate_fn과 동일)
# slot 순서: precedingId, followingId,
#            leftPrecedingId, leftAlongsideId, leftFollowingId,
#            rightPrecedingId, rightAlongsideId, rightFollowingId
SLOT_LANE_ROW = {
    0: 1,   # preceding      → 같은 차선 앞
    1: 1,   # following      → 같은 차선 뒤
    2: 0,   # leftPreceding  → 왼쪽 차선 앞
    3: 0,   # leftAlongside  → 왼쪽 차선 옆
    4: 0,   # leftFollowing  → 왼쪽 차선 뒤
    5: 2,   # rightPreceding → 오른쪽 차선 앞
    6: 2,   # rightAlongside → 오른쪽 차선 옆
    7: 2,   # rightFollowing → 오른쪽 차선 뒤
}


def _dx_to_bin(dx: float) -> int:
    """dx (ft) → 13-bin 인덱스"""
    return int(np.argmin(np.abs(BIN_CENTERS - dx)))


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class ngsimDataset(Dataset):
    """
    새 전처리(preprocess.py)가 생성한 mmap 파일을 읽어
    기존 model.py(CS-LSTM)가 기대하는 포맷으로 변환한다.

    x_nb 특징 인덱스 (neighformer preprocess.py 스키마, 13 dims):
        0 dx   1 dy   2 dvx   3 dvy   4 dax   5 day
        6 lc_state   7 lit   8 lis   9 gate   10 I_x   11 I_y   12 I

    nbr_feature_mode:
        0 → (dx, dy)                              dim=2  [baseline]
        1 → (dax, day, lc_state, dx_time, gate)   dim=5
        2 → (dx, dy, dvx, dvy, dax, day, gate)    dim=7
        3 → (lc_state, dx_time)                   dim=2
        4 → all 13 dims
        5 → (dx, dy, dvx, dvy, dax, day)          dim=6  [c0]
        6 → (dx, dy, dvx, dvy, dax, day, I_y)     dim=7  [c2]
        7 → (dx, dy, dvx, dvy, dax, day, I)       dim=7  [c1]
        8 → (dx, dy, dvx, dvy, dax, day, lis, I_y)     dim=8  [c3]
    """

    def __init__(self, mmap_dir, t_h=None, t_f=None, d_s=None,
                 enc_size=64, grid_size=(13, 3), nbr_feature_mode=0,
                 indices=None):
        mmap_dir = Path(mmap_dir)

        self.x_ego   = np.load(mmap_dir / "x_ego.npy",   mmap_mode='r')  # (N, T, 6)
        self.y       = np.load(mmap_dir / "y.npy",       mmap_mode='r')  # (N, Tf, 2)
        self.x_nb    = np.load(mmap_dir / "x_nb.npy",    mmap_mode='r')  # (N, T, 8, 13)
        self.nb_mask = np.load(mmap_dir / "nb_mask.npy", mmap_mode='r')  # (N, T, 8) bool

        self.nbr_feature_mode = nbr_feature_mode
        self.enc_size         = enc_size
        self.grid_size        = grid_size  # (13, 3) — model 호환용

        self.T  = self.x_ego.shape[1]
        self.Tf = self.y.shape[1]

        N = len(self.x_ego)
        self.indices = np.asarray(indices) if indices is not None else np.arange(N)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])

        # ego: (x, y) relative coords, shape (T, 2)
        ego_xy = torch.from_numpy(self.x_ego[idx, :, 0:2].copy()).float()

        # future: (Tf, 2)
        fut = torch.from_numpy(self.y[idx].copy()).float()

        # neighbors: feature 선택 후 (T, 8, D)
        nbr_raw = self.x_nb[idx]    # (T, 8, 13)
        nb_mask = self.nb_mask[idx] # (T, 8) bool

        if self.nbr_feature_mode == 0:
            nbr_feat = nbr_raw[:, :, 0:2]
        elif self.nbr_feature_mode == 1:
            nbr_feat = nbr_raw[:, :, 4:9]
        elif self.nbr_feature_mode == 2:
            nbr_feat = nbr_raw[:, :, [0, 1, 2, 3, 4, 5, 8]]
        elif self.nbr_feature_mode == 3:
            nbr_feat = nbr_raw[:, :, 6:8]
        elif self.nbr_feature_mode == 4:
            nbr_feat = nbr_raw[:, :, 0:13]
        elif self.nbr_feature_mode == 5:
            # c0: (dx, dy, dvx, dvy, dax, day)
            nbr_feat = nbr_raw[:, :, 0:6]
        elif self.nbr_feature_mode == 6:
            # c2: (dx, dy, dvx, dvy, dax, day, I_y)  — idx 11 = I_y
            nbr_feat = nbr_raw[:, :, [0, 1, 2, 3, 4, 5, 11]]
        elif self.nbr_feature_mode == 7:
            # c2: (dx, dy, dvx, dvy, dax, day, I)
            nbr_feat = nbr_raw[:, :, [0, 1, 2, 3, 4, 5, 12]]
        elif self.nbr_feature_mode == 8:
            # c3: (dx, dy, dvx, dvy, dax, day, lis, I_y)
            nbr_feat = nbr_raw[:, :, [0, 1, 2, 3, 4, 5, 8, 11]]
        else:
            nbr_feat = nbr_raw[:, :, 0:2]

        nbr_feat = torch.from_numpy(nbr_feat.copy()).float()   # (T, 8, D)
        nb_mask  = torch.from_numpy(nb_mask.copy())            # (T, 8) bool

        # dx (index 0) 는 collate_fn에서 grid bin 결정에 필요
        dx_raw = torch.from_numpy(nbr_raw[:, :, 0].copy()).float()  # (T, 8)

        # maneuver labels: 새 파이프라인에 없으므로 dummy (class index 1 → one-hot index 0)
        lat_enc = torch.tensor(1, dtype=torch.long)
        lon_enc = torch.tensor(1, dtype=torch.long)

        return ego_xy, fut, nbr_feat, nb_mask, dx_raw, lat_enc, lon_enc

    # ------------------------------------------------------------------
    def collate_fn(self, samples):
        """
        기존 model.py가 기대하는 출력을 그대로 유지:
            hist_batch     (T, B, 2)
            nbrs_batch     (T, N_active, D)
            soc_mask_batch (B, 3, 13, enc_size)   ← 13×3 그리드
            lat_enc_batch  (B, 3)
            lon_enc_batch  (B, 2)
            fut_batch      (Tf, B, 2)
            op_mask_batch  (Tf, B, 2)

        8 슬롯 → 13×3 그리드 변환:
            - lane_row : SLOT_LANE_ROW 테이블로 고정
            - bin_x    : 유효한 마지막 타임스텝의 dx 값으로 결정
            - 충돌(같은 셀에 두 슬롯)이 나면 먼저 처리된 슬롯 유지
              (기존 preprocess_highd.py의 best_d 처리와 동등)
        """
        batch_size = len(samples)

        hist_batch = torch.stack([s[0] for s in samples], dim=1)  # (T, B, 2)
        fut_batch  = torch.stack([s[1] for s in samples], dim=1)  # (Tf, B, 2)
        T  = hist_batch.shape[0]
        Tf = fut_batch.shape[0]

        op_mask_batch = torch.ones(Tf, batch_size, 2)

        # maneuver one-hot
        lat_enc_batch = torch.zeros(batch_size, 3)
        lon_enc_batch = torch.zeros(batch_size, 2)
        for i, s in enumerate(samples):
            lat_idx = max(0, min(int(s[5].item()) - 1, 2))
            lon_idx = max(0, min(int(s[6].item()) - 1, 1))
            lat_enc_batch[i, lat_idx] = 1
            lon_enc_batch[i, lon_idx] = 1

        # 13×3 소셜 마스크 (B, GRID_Y=3, GRID_X=13, enc_size)
        soc_mask_batch = torch.zeros(batch_size, GRID_Y, GRID_X,
                                     self.enc_size, dtype=torch.bool)
        nbr_seqs  = []
        input_dim = samples[0][2].shape[-1]

        for b_i, s in enumerate(samples):
            nbr_feat = s[2]   # (T, 8, D)
            nb_mask  = s[3]   # (T, 8) bool
            dx_raw   = s[4]   # (T, 8)

            for slot in range(8):
                if not nb_mask[:, slot].any():
                    continue

                # 대표 dx: 유효한 마지막 타임스텝 사용
                valid_t = nb_mask[:, slot].nonzero(as_tuple=False).flatten()
                rep_dx  = float(dx_raw[valid_t[-1], slot].item())

                gy = SLOT_LANE_ROW[slot]
                gx = _dx_to_bin(rep_dx)

                # 셀 충돌이면 skip (먼저 온 슬롯 우선)
                if soc_mask_batch[b_i, gy, gx, 0]:
                    continue

                nbr_seqs.append(nbr_feat[:, slot, :])   # (T, D)
                soc_mask_batch[b_i, gy, gx, :] = True

        if len(nbr_seqs) > 0:
            nbrs_batch = torch.stack(nbr_seqs, dim=1)   # (T, N_active, D)
        else:
            nbrs_batch = torch.zeros(T, 0, input_dim)

        return (hist_batch, nbrs_batch, soc_mask_batch,
                lat_enc_batch, lon_enc_batch,
                fut_batch, op_mask_batch)


# -------------------------------------------------------------------------
# outputActivation & loss utilities  (원본과 동일)
# -------------------------------------------------------------------------

def outputActivation(x):
    muX  = x[:, :, 0:1];  muY  = x[:, :, 1:2]
    sigX = x[:, :, 2:3];  sigY = x[:, :, 3:4]
    rho  = x[:, :, 4:5]
    sigX = torch.exp(sigX);  sigY = torch.exp(sigY)
    rho  = torch.tanh(rho)
    return torch.cat([muX, muY, sigX, sigY, rho], dim=2)


def maskedNLL(y_pred, y_gt, mask):
    acc  = torch.zeros_like(mask)
    muX  = y_pred[:, :, 0];  muY  = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2];  sigY = y_pred[:, :, 3]
    rho  = y_pred[:, :, 4]
    ohr  = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x    = y_gt[:, :, 0];    y    = y_gt[:, :, 1]
    out  = (0.5 * torch.pow(ohr, 2) *
            (torch.pow(sigX, 2) * torch.pow(x - muX, 2) +
             torch.pow(sigY, 2) * torch.pow(y - muY, 2) -
             2 * rho * sigX * sigY * (x - muX) * (y - muY))
            - torch.log(sigX * sigY * ohr) + 1.8379)
    acc[:, :, 0] = out;  acc[:, :, 1] = out
    acc = acc * mask
    return torch.sum(acc) / torch.sum(mask)


def maskedMSE(y_pred, y_gt, mask):
    acc  = torch.zeros_like(mask)
    muX  = y_pred[:, :, 0];  muY  = y_pred[:, :, 1]
    x    = y_gt[:, :, 0];    y    = y_gt[:, :, 1]
    out  = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out;  acc[:, :, 1] = out
    acc = acc * mask
    return torch.sum(acc) / torch.sum(mask)


def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False):
    if use_maneuvers:
        acc   = torch.zeros(op_mask.shape[0], op_mask.shape[1],
                            num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                w      = lat_pred[:, l] * lon_pred[:, k]
                w      = w.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[count]
                muX    = y_pred[:, :, 0];  muY  = y_pred[:, :, 1]
                sigX   = y_pred[:, :, 2];  sigY = y_pred[:, :, 3]
                rho    = y_pred[:, :, 4]
                ohr    = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x      = fut[:, :, 0];     y    = fut[:, :, 1]
                out    = -(0.5 * torch.pow(ohr, 2) *
                           (torch.pow(sigX, 2) * torch.pow(x - muX, 2) +
                            torch.pow(sigY, 2) * torch.pow(y - muY, 2) -
                            2 * rho * sigX * sigY * (x - muX) * (y - muY))
                           - torch.log(sigX * sigY * ohr) + 1.8379)
                acc[:, :, count] = out + torch.log(w)
                count += 1
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            return torch.sum(acc) / torch.sum(op_mask[:, :, 0])
        else:
            return torch.sum(acc, dim=0), torch.sum(op_mask[:, :, 0], dim=0)
    else:
        y_pred = fut_pred
        muX    = y_pred[:, :, 0];  muY  = y_pred[:, :, 1]
        sigX   = y_pred[:, :, 2];  sigY = y_pred[:, :, 3]
        rho    = y_pred[:, :, 4]
        ohr    = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x      = fut[:, :, 0];     y    = fut[:, :, 1]
        acc    = (0.5 * torch.pow(ohr, 2) *
                  (torch.pow(sigX, 2) * torch.pow(x - muX, 2) +
                   torch.pow(sigY, 2) * torch.pow(y - muY, 2) -
                   2 * rho * sigX * sigY * (x - muX) * (y - muY))
                  - torch.log(sigX * sigY * ohr) + 1.8379)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            return torch.sum(acc) / torch.sum(op_mask[:, :, 0])
        else:
            return torch.sum(acc, dim=0), torch.sum(op_mask[:, :, 0], dim=0)


def maskedMSETest(y_pred, y_gt, mask):
    acc  = torch.zeros_like(mask)
    muX  = y_pred[:, :, 0];  muY = y_pred[:, :, 1]
    x    = y_gt[:, :, 0];    y   = y_gt[:, :, 1]
    out  = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out;  acc[:, :, 1] = out
    acc = acc * mask
    return torch.sum(acc[:, :, 0], dim=0), torch.sum(mask[:, :, 0], dim=0)


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs