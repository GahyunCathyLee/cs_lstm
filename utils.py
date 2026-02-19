from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import h5py

# -------------------------------------------------------------------------
# Dataset class for the NGSIM/HighD dataset (Optimized for Pre-calculated Tensor)
# -------------------------------------------------------------------------
class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13,3), nbr_feature_mode=0):
        # 파일 확장자에 따라 로드 방식 분기 (mat vs h5)
        if str(mat_file).endswith('.h5'):
            self.data = {}
            with h5py.File(mat_file, 'r') as f:
                for k in f.keys():
                    self.data[k] = f[k][:]
        else:
            self.data = scp.loadmat(mat_file)
        
        # 1. 텐서 로드 (전처리된 Full Data)
        self.hist = torch.from_numpy(self.data['hist']).float()
        self.fut = torch.from_numpy(self.data['fut']).float()
        self.nbrs = torch.from_numpy(self.data['nbrs']).float() # Shape: (N, 39, T, 9)
        self.lat_enc = torch.from_numpy(self.data['lat_enc']).long()
        self.lon_enc = torch.from_numpy(self.data['lon_enc']).long()
        self.op_mask = torch.from_numpy(self.data['op_mask']).float()
        
        # 2. 설정 저장
        self.nbr_feature_mode = nbr_feature_mode
        self.enc_size = enc_size 
        self.d_s = d_s
        self.grid_size = grid_size

    def __len__(self):
        return len(self.hist) # [수정됨] self.D 대신 self.hist 사용

    def __getitem__(self, idx):
        hist = self.hist[idx] 
        fut = self.fut[idx]
        nbr_data = self.nbrs[idx] # (39, T, 9)
        
        # ---------------------------------------------------------
        # 실험 모드에 따른 Feature Slicing
        # ---------------------------------------------------------
        if self.nbr_feature_mode == 0: 
            # [기존 버전] (x, y)
            sel_nbrs = nbr_data[:, :, 0:2] 
            
        elif self.nbr_feature_mode == 1: 
            # [실험 1] (ax, ay, lc, dxt, gate)
            sel_nbrs = nbr_data[:, :, 4:9]
            
        elif self.nbr_feature_mode == 2: 
            # [실험 2] (x, y, vx, vy, ax, ay, gate)
            idx_list = torch.tensor([0, 1, 2, 3, 4, 5, 8]).long()
            sel_nbrs = nbr_data[:, :, idx_list]
            
        elif self.nbr_feature_mode == 3: 
            # [실험 3] (lc, dxt)
            sel_nbrs = nbr_data[:, :, 6:8]
            
        else:
            sel_nbrs = nbr_data[:, :, 0:2]

        return hist, fut, sel_nbrs, self.lat_enc[idx], self.lon_enc[idx]

    ## Collate function for dataloader
    def collate_fn(self, samples):
        batch_size = len(samples)
        
        # 1. Stack Basic Tensors
        hist_batch = torch.stack([s[0] for s in samples], dim=1) # (T, Batch, 2)
        fut_batch = torch.stack([s[1] for s in samples], dim=1)  # (T_f, Batch, 2)
        op_mask_batch = torch.ones(fut_batch.shape[0], batch_size, 2)
        
        lat_enc_batch = torch.zeros(batch_size, 3) # Lat Classes = 3
        lon_enc_batch = torch.zeros(batch_size, 2) # Lon Classes = 2
        
        for i, s in enumerate(samples):
            # s[3]: lat_enc (1, 2, 3) -> index (0, 1, 2)
            # s[4]: lon_enc (1, 2)    -> index (0, 1)
            
            # 값 가져오기 (.item() 사용)
            lat_val = int(s[3].item())
            lon_val = int(s[4].item())
            
            # 1-based index (MATLAB) -> 0-based index (Python) 변환
            lat_idx = max(0, min(lat_val - 1, 2)) # 0~2 사이로 클램핑
            lon_idx = max(0, min(lon_val - 1, 1)) # 0~1 사이로 클램핑
            
            lat_enc_batch[i, lat_idx] = 1
            lon_enc_batch[i, lon_idx] = 1

        # 2. Process Neighbors & Social Mask
        soc_mask_batch = torch.zeros(batch_size, 3, 13, self.enc_size, dtype=torch.bool)
        nbr_seqs = []
        
        # 입력 차원 감지
        input_dim = samples[0][2].shape[-1]
        t_len = samples[0][2].shape[1]
        
        for b_i, (_, _, nbr_tensor, _, _) in enumerate(samples):
            for grid_id in range(39):
                if torch.sum(torch.abs(nbr_tensor[grid_id])) > 0:
                    nbr_seqs.append(nbr_tensor[grid_id]) 
                    gx = grid_id % 13
                    gy = grid_id // 13
                    soc_mask_batch[b_i, gy, gx, :] = True
        
        if len(nbr_seqs) > 0:
            nbrs_batch = torch.stack(nbr_seqs, dim=1) 
        else:
            nbrs_batch = torch.zeros(t_len, 0, input_dim)

        return hist_batch, nbrs_batch, soc_mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch

## (Evaluation용 Helper 함수들은 그대로 둠)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True, avg_along_time=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                w = lat_pred[:,l]*lon_pred[:,k]
                w = w.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[count]
                muX = y_pred[:,:,0]
                muY = y_pred[:,:,1]
                sigX = y_pred[:,:,2]
                sigY = y_pred[:,:,3]
                rho = y_pred[:,:,4]
                ohr = torch.pow(1-torch.pow(rho,2),-0.5)
                x = fut[:,:, 0]
                y = fut[:,:, 1]
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                acc[:,:,count] =  out + torch.log(w)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = fut[:, :, 0]
        y = fut[:, :, 1]
        out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc, dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts

def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs