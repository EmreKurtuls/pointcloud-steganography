"""
Clean fine-tune of the message decoder (Google Colab).

Freezes the embedding network and retrains only `msg_decoder` on clean
(no-noise) stego clouds. The released models/model.pth is the output of this
step (v5bch_clean_ft.pth).

NOTE: This was run as a notebook cell *after* training/train.py in the same
Colab session. It relies on names defined there (StegoV5BCH, device,
bch_encode_batch_tensor, encode_message_128, CHUNK_BITS, MODEL_BITS,
TOTAL_PAYLOAD). Run train.py's definitions first (with train() not called),
or paste both into one notebook.
"""

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

FINETUNE_CKPT = 'v5bch_best_63b.pth'
SAVE_CKPT     = 'v5bch_clean_ft.pth'
EVAL_ALPHA_FT = 0.08
LR_FT         = 8e-4
MAX_FT_EPOCHS = 100
BATCH_SIZE_FT = 512
LAMBDA_MSG_FT = 200
PATIENCE      = 4
SAVE_PATH     = '/content/drive/MyDrive/pointcloud_stego/ModelNet40'

print(f"\n{'='*60}")
print(f"  MSG_DECODER CLEAN FINE-TUNE")
print(f"  No noise | LAMBDA_MSG={LAMBDA_MSG_FT} | LR={LR_FT}")
print(f"  Early stopping: patience={PATIENCE} eval checks")
print(f"  Source: {FINETUNE_CKPT} → Output: {SAVE_CKPT}")
print(f"{'='*60}\n")

ft_model = StegoV5BCH().to(device)
ft_model.load_state_dict(
    torch.load(os.path.join(SAVE_PATH, FINETUNE_CKPT), map_location=device))
ft_model.train()

frozen = trainable = 0
for name, param in ft_model.named_parameters():
    if 'msg_decoder' in name:
        param.requires_grad = True;  trainable += 1
    else:
        param.requires_grad = False; frozen += 1

n_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
print(f"✓ Frozen: {frozen} | Trainable: {trainable} | Params: {n_params:,}")


class SimpleDS(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pc = self.data[idx, :4096].clone()
        pc = pc - pc.mean(0)
        pc = pc / (torch.sqrt((pc**2).sum(1)).max() + 1e-8)
        if random.random() > 0.5:
            theta = np.random.uniform(0, 2*np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R     = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32)
            pc    = pc @ R.T
        return pc

train_ds_ft = SimpleDS(os.path.join(SAVE_PATH, 'train4096_HD.pt'))
test_ds_ft  = SimpleDS(os.path.join(SAVE_PATH, 'test4096_HD.pt'))
train_loader_ft = DataLoader(train_ds_ft, BATCH_SIZE_FT, shuffle=True,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)
print(f"✓ Train: {len(train_ds_ft):,} | Test: {len(test_ds_ft):,}")

optimizer = torch.optim.AdamW(
    [p for p in ft_model.parameters() if p.requires_grad],
    lr=LR_FT, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_FT_EPOCHS, eta_min=1e-6)

scaler_ft = torch.amp.GradScaler('cuda')

def quick_eval(model, n=300):
    test_data = test_ds_ft.data
    B = 64; ch_accs = []
    for start in range(0, n, B):
        end = min(start+B, n); actual = end-start
        pcs, chs = [], []
        for i in range(actual):
            msg = [random.randint(0,1) for _ in range(TOTAL_PAYLOAD)]
            ch  = encode_message_128(msg)[0]
            pc  = test_data[(start+i) % len(test_data)]
            pc  = pc - pc.mean(0)
            pc  = pc / (torch.sqrt((pc**2).sum(1)).max() + 1e-8)
            pcs.append(pc); chs.append(ch)
        with torch.no_grad():
            pc_t   = torch.stack(pcs).to(device)
            ch_t   = torch.tensor(chs, dtype=torch.float32, device=device)
            stego  = model.embed(pc_t, ch_t, EVAL_ALPHA_FT)
            logits = model.extract(stego)
            pred   = (logits > 0).int().cpu()
            acc    = (pred == torch.tensor(chs)).float().mean().item()
            ch_accs.append(acc)
    return np.mean(ch_accs)

baseline = quick_eval(ft_model)
print(f"\nBaseline channel accuracy: {baseline*100:.1f}%\n")

best_acc     = baseline
best_epoch   = -1
patience_cnt = 0

for epoch in range(MAX_FT_EPOCHS):
    ft_model.train()
    t_acc = t_loss = 0; n = 0

    pbar = tqdm(train_loader_ft,
                desc=f"Ep{epoch:03d}|LR={scheduler.get_last_lr()[0]:.2e}",
                leave=False)

    for pc in pbar:
        pc  = pc.to(device, non_blocking=True)
        B   = pc.shape[0]
        optimizer.zero_grad(set_to_none=True)

        data   = torch.randint(0, 2, (B, CHUNK_BITS), dtype=torch.float32)
        target = bch_encode_batch_tensor(data).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                stego = ft_model.embed(pc, target, EVAL_ALPHA_FT)

            logits   = ft_model.extract(stego)
            loss_msg = F.binary_cross_entropy_with_logits(
                           logits[:, :MODEL_BITS], target[:, :MODEL_BITS])
            loss     = LAMBDA_MSG_FT * loss_msg

        scaler_ft.scale(loss).backward()
        scaler_ft.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in ft_model.parameters() if p.requires_grad], 1.0)
        scaler_ft.step(optimizer); scaler_ft.update()

        pred = (logits[:, :MODEL_BITS] > 0).float()
        acc  = (pred == target[:, :MODEL_BITS]).float().mean().item()
        t_acc += acc; t_loss += loss_msg.item(); n += 1
        pbar.set_postfix({'Acc': f'{acc:.3f}'})

    scheduler.step()
    avg_acc = t_acc / n

    if epoch % 3 == 0 or epoch == MAX_FT_EPOCHS - 1:
        eval_acc = quick_eval(ft_model)
        delta    = eval_acc - baseline
        sign     = f"▲ +{delta*100:.1f}%" if delta > 0 else f"▼ {delta*100:.1f}%"
        print(f"Ep{epoch:03d}| train={avg_acc*100:.1f}% "
              f"eval={eval_acc*100:.1f}% {sign} vs baseline")

        if eval_acc > best_acc:
            best_acc     = eval_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(ft_model.state_dict(),
                       os.path.join(SAVE_PATH, SAVE_CKPT))
            print(f"  ★ Saved {SAVE_CKPT} ({best_acc*100:.1f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stop: no improvement for "
                      f"{PATIENCE} eval checks")
                break
    else:
        print(f"Ep{epoch:03d}| train={avg_acc*100:.1f}%")

print(f"\n{'='*60}")
print(f"  Baseline:   {baseline*100:.1f}%")
print(f"  Best eval:  {best_acc*100:.1f}% at epoch {best_epoch}")
print(f"  Improvement: +{(best_acc-baseline)*100:.1f}%")
print(f"  Saved: {SAVE_CKPT}")
print(f"  Original {FINETUNE_CKPT} untouched ")
print(f"{'='*60}")

