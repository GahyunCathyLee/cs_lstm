from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSE, maskedNLLTest
from torch.utils.data import DataLoader
import time
import math
import yaml         
import argparse     
import os
from pathlib import Path
import shutil
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Train CS-LSTM model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = parse_args()
    
    args = config['model_args']
    args['train_flag'] = True 

    t_args = config['train_args']
    ds_args = config.get('dataset_args', {})
    paths = config['data_paths']

    nbr_mode = ds_args.get('nbr_mode', 0)
    mode_dim_map = {
        0: 2, # Original (x, y)
        1: 5, # (ax, ay, lc, dxt, gate)
        2: 7, # (x, y, vx, vy, ax, ay, gate)
        3: 2  # (lc, dxt)
    }
    current_nbr_dim = mode_dim_map.get(nbr_mode, 2)
    args['nbr_input_dim'] = current_nbr_dim
    print(f"✅ Experiment Mode: {nbr_mode} | Neighbor Input Dim: {current_nbr_dim}")

    # -----------------------------
    # Save path / directory setup
    # -----------------------------
    save_dir = Path(paths['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt_path = save_dir / "last.pt"
    best_ckpt_path = save_dir / "best.pt"

    # Initialize network
    net = highwayNet(args)
    if args['use_cuda']:
        net = net.cuda()
        device = torch.device("cuda")

    ## Initialize optimizer
    pretrainEpochs = t_args['pretrain_epochs']
    trainEpochs = t_args['train_epochs']
    batch_size = t_args['batch_size']
    optimizer = torch.optim.Adam(net.parameters())
    crossEnt = torch.nn.BCELoss()

    ## Initialize data loaders
    trSet  = ngsimDataset(paths['train_set'],
                        t_h=ds_args.get('t_h', 15),
                        t_f=ds_args.get('t_f', 25),
                        d_s=ds_args.get('d_s', 1),
                        enc_size=args['encoder_size'],
                        grid_size=tuple(args['grid_size']),
                        nbr_feature_mode=nbr_mode)
    valSet = ngsimDataset(paths['val_set'],
                        t_h=ds_args.get('t_h', 15),
                        t_f=ds_args.get('t_f', 25),
                        d_s=ds_args.get('d_s', 1),
                        enc_size=args['encoder_size'],
                        grid_size=tuple(args['grid_size']),
                        nbr_feature_mode=nbr_mode)
    trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=valSet.collate_fn)

    ## Variables holding train and validation loss values:
    train_loss = []
    val_loss = []
    best_val_loss = math.inf

    for epoch_num in range(pretrainEpochs+trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('\nTraining with NLL loss')

        print(f"\n===== Epoch {epoch_num+1}/{pretrainEpochs+trainEpochs} =====")
        epoch_st_time = time.time()
        ## Train:
        net.train_flag = True

        # Variables to track training performance:
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0

        for i, data in enumerate(trDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            if args['use_cuda']:
                hist = hist.to(device, non_blocking=True)
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            # Forward pass
            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                    avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            # Backprop and update weights
            optimizer.zero_grad()
            l.backward()
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Track average train loss and time
            batch_time = time.time() - st_time
            avg_tr_loss += l.item()
            avg_tr_time += batch_time

            if i % 100 == 99:
                eta = avg_tr_time / 100 * (len(trDataloader) - i)
                eta_m, eta_s = divmod(int(eta), 60)
                elapsed_sec = int(time.time() - epoch_st_time)
                elapsed_m, elapsed_s = divmod(elapsed_sec, 60)

                msg = f"[Train] Loss: {avg_tr_loss / 100:.4f}, Acc(lon lat): {avg_lat_acc / 100:.4f} {avg_lon_acc / 100:.4f} | Progress: {i / len(trDataloader) * 100:.2f}% [{elapsed_m:02d}:{elapsed_s:02d}<{eta_m:02d}:{eta_s:02d}]"
                print(msg + " " * 5, end='\r')
                sys.stdout.flush()

                train_loss.append(avg_tr_loss / 100)
                avg_tr_loss = 0
                avg_lat_acc = 0
                avg_lon_acc = 0
                avg_tr_time = 0

        elapsed_sec = int(time.time() - epoch_st_time)
        elapsed_m, elapsed_s = divmod(elapsed_sec, 60)        
        msg = f"[Train] Loss: {avg_tr_loss / 100:.4f}, Acc(lon lat): {avg_lat_acc / 100:.4f} {avg_lon_acc / 100:.4f} | Progress: 100% [{elapsed_m:02d}:{elapsed_s:02d}]"
        print(msg + " " * 10, end='\r')
        print()

        ## Validate:
        net.train_flag = False

        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0

        for i, data  in enumerate(valDataloader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            # Forward pass
            if args['use_maneuvers']:
                if epoch_num < pretrainEpochs:
                    net.train_flag = True
                    fut_pred, _ , _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, avg_along_time=True)
                    avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        # Print validation loss and update display variables
        current_val_loss = avg_val_loss / val_batch_count
        print(f'[Valid] Loss: {current_val_loss:.4f}, Acc(lon lat):',
              format(avg_val_lat_acc / val_batch_count * 100, '0.4f'),
              format(avg_val_lon_acc / val_batch_count * 100, '0.4f'))
        
        val_loss.append(current_val_loss)
        prev_val_loss = current_val_loss

        # -----------------------------
        # Checkpoint saving
        # -----------------------------
        # 1) last
        torch.save(net.state_dict(), last_ckpt_path)

        # 2) best
        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss
            torch.save(net.state_dict(), best_ckpt_path)
            print(f"[CKPT] ✅ Best updated @epoch {epoch_num+1} (val_loss={best_val_loss:.4f}) -> {best_ckpt_path}")

        # 3) every 100 epochs
        epoch_1based = epoch_num + 1
        if epoch_1based % 100 == 0:
            snap_path = save_dir / f"{epoch_1based}_best.pt"
            if best_ckpt_path.exists():
                shutil.copy2(best_ckpt_path, snap_path)
            else:
                shutil.copy2(last_ckpt_path, snap_path)
            print(f"[INTERVAL] 💾 Saved best model up to epoch {epoch_1based} -> {snap_path}")


if __name__ == '__main__':
    main()