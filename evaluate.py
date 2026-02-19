from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import yaml         
import argparse     
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CS-LSTM model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = parse_args()
    args = config['model_args']
    args['train_flag'] = False 
    paths = config['data_paths']
    batch_size = config['train_args']['batch_size']

    nllLoss = torch.zeros(25)
    mseLoss = torch.zeros(25)
    counts = torch.zeros(25)

    # Initialize network
    net = highwayNet(args)
    
    device = torch.device("cuda" if args['use_cuda'] and torch.cuda.is_available() else "cpu")
    ckpt_path = paths['save_dir'] + "/best.pt"
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    if args['use_cuda']:
        net = net.cuda()

    ds_args = config.get('dataset_args', {})
    tsSet = ngsimDataset(paths['test_set'],
                        t_h=ds_args.get('t_h', 15),
                        t_f=ds_args.get('t_f', 25),
                        d_s=ds_args.get('d_s', 1),
                        enc_size=args['encoder_size'],
                        grid_size=tuple(args['grid_size']))
    tsDataloader = DataLoader(tsSet, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

    if args['use_cuda']:
        nllLoss = nllLoss.cuda()
        mseLoss = mseLoss.cuda()
        counts = counts.cuda()

    ade_total = 0.0
    fde_total = 0.0
    total_samples = 0

    eval_loop = tqdm(tsDataloader, desc="Evaluating")

    for i, data in enumerate(eval_loop):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        if args['use_cuda']:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = \
                hist.cuda(), nbrs.cuda(), mask.cuda(), lat_enc.cuda(), lon_enc.cuda(), fut.cuda(), op_mask.cuda()

        with torch.no_grad(): 
            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                l_nll, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
                
                fut_pred_tensor = torch.stack(fut_pred) 

                lat_man = torch.argmax(lat_pred, dim=1)
                lon_man = torch.argmax(lon_pred, dim=1)
                indx = lon_man * 3 + lat_man  # (batch,)

                batch_range = torch.arange(fut.shape[1], device=fut.device)

                fut_pred_max = fut_pred_tensor[indx, :, batch_range, :].transpose(0, 1)
            else:
                fut_pred_max = net(hist, nbrs, mask, lat_enc, lon_enc)
                l_nll, c = maskedNLLTest(fut_pred_max, 0, 0, fut, op_mask, use_maneuvers=False)
            
            l_mse, _ = maskedMSETest(fut_pred_max, fut, op_mask)

            error = torch.norm(fut_pred_max[:, :, :2] - fut, p=2, dim=2) * 0.3048
            mask_flat = op_mask[:, :, 0] 

            for b in range(fut.shape[1]): 
                valid_idx = torch.where(mask_flat[:, b] > 0)[0]
                if len(valid_idx) > 0:
                    ade_total += error[valid_idx, b].mean().item()
                    fde_total += error[valid_idx[-1], b].item()
                    total_samples += 1

        nllLoss += l_nll.detach()
        mseLoss += l_mse.detach()
        counts += c.detach()

    valid_mask = counts > 0
    avg_nll = (nllLoss[valid_mask] / counts[valid_mask]).mean().item()
    
    mse_per_frame = mseLoss[valid_mask] / counts[valid_mask]
    rmse_per_frame = torch.pow(mse_per_frame, 0.5) * 0.3048
    avg_rmse = rmse_per_frame.mean().item()

    avg_ade = ade_total / total_samples if total_samples > 0 else 0
    avg_fde = fde_total / total_samples if total_samples > 0 else 0

    print(f"\nFinal Evaluation Results:")
    print(f"Average NLL: {avg_nll:.4f}")
    print(f"Average RMSE (Meters): {avg_rmse:.4f}")

    indices = [4, 9, 14, 19, 24] 
    rmse_outputs = [f"@{i+1}s: {rmse_per_frame[idx].item():.4f}" for i, idx in enumerate(indices) if idx < len(rmse_per_frame)]
    print("RMSE: " + ",  ".join(rmse_outputs))

    print(f"Average ADE: {avg_ade:.4f}m")
    print(f"Average FDE: {avg_fde:.4f}m")

if __name__ == '__main__':
    main()