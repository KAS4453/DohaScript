"""
main_run.py — DohaScript GANwriting training script.

Usage:
    python main_run.py 0          # train from scratch
    python main_run.py 1000       # resume from epoch 1000
"""

import os

# Reduce CUDA memory fragmentation (helps with OOM on Blackwell/large models)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import glob
import time
import argparse

import numpy as np
import torch
from torch import optim

from load_data import NUM_WRITERS, loadData
from network_tro import ConTranModel
from loss_tro import CER
from metrics_logger import MetricsLogger, ImageAccumulator, compute_metrics

# ─── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='resume from this epoch; 0=scratch; -1=auto-resume from latest checkpoint')
args = parser.parse_args()

# ─── Config ──────────────────────────────────────────────────────────────────
gpu = torch.device('cuda')
OOV = True          # True: sample random OOV words from Devanagari corpus

NUM_THREAD       = 4
BATCH_SIZE       = 4        # raise back to 8 once GPU is free
EVAL_EPOCH       = 20
MODEL_SAVE_EPOCH = 20       # checkpoint every eval epoch — never lose more than 20 epochs
SHOW_ITER_NUM    = 500
EARLY_STOP_EPOCH = None     # set to int (e.g. 200) to enable early stopping
KEEP_LAST_N      = 5        # only keep the 5 most recent checkpoints to save disk space

# Discriminator was crushing the generator (dis<tr>=0.17 vs random 0.693 at ep1440).
# Fix: slow dis down, speed gen/rec/cla up so all four components stay competitive.
lr_dis = 2e-5   # was 1e-4; reduce to stop dis dominating
lr_gen = 1e-4   # unchanged
lr_rec = 5e-5   # was 1e-5; increase so recognizer signal propagates faster
lr_cla = 5e-5   # was 1e-5; increase so writer style signal reaches generator

if args.start_epoch == -1:
    # Auto-resume: find the latest checkpoint in save_weights/
    import glob as _glob
    ckpts = sorted(_glob.glob('save_weights/contran-*.model'),
                   key=lambda p: int(p.split('-')[1].split('.')[0]))
    if ckpts:
        CurriculumModelID = int(ckpts[-1].split('-')[1].split('.')[0])
        print(f'[auto-resume] found checkpoint → resuming from epoch {CurriculumModelID}')
    else:
        CurriculumModelID = 0
        print('[auto-resume] no checkpoints found → starting from scratch')
else:
    CurriculumModelID = args.start_epoch


# ─── Data ────────────────────────────────────────────────────────────────────

def sort_batch(batch):
    domains, wids, idxs_list = [], [], []
    imgs, img_widths, labels = [], [], []
    img_xts, label_xts, label_xts_swap = [], [], []

    for (domain, wid, idx, img, img_width, label,
         img_xt, label_xt, label_xt_swap) in batch:
        domains.append(domain)
        wids.append(wid)
        idxs_list.append(idx)
        imgs.append(img)
        img_widths.append(img_width)
        labels.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    wids           = torch.from_numpy(np.array(wids,           dtype='int64'))
    imgs           = torch.from_numpy(np.array(imgs,           dtype='float32'))
    img_widths     = torch.from_numpy(np.array(img_widths,     dtype='int64'))
    labels         = torch.from_numpy(np.array(labels,         dtype='int64'))
    img_xts        = torch.from_numpy(np.array(img_xts,        dtype='float32'))
    label_xts      = torch.from_numpy(np.array(label_xts,      dtype='int64'))
    label_xts_swap = torch.from_numpy(np.array(label_xts_swap, dtype='int64'))

    return (np.array(domains), wids, idxs_list, imgs, img_widths, labels,
            img_xts, label_xts, label_xts_swap)


def all_data_loader():
    data_train, data_test = loadData(OOV)
    train_loader = torch.utils.data.DataLoader(
        data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=NUM_THREAD, pin_memory=True,
        drop_last=True)   # drop last batch if < BATCH_SIZE (BatchNorm needs >1)
    test_loader  = torch.utils.data.DataLoader(
        data_test,  collate_fn=sort_batch, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_THREAD, pin_memory=True,
        drop_last=False)  # keep all test batches; eval uses model.eval() so BN is fine
    return train_loader, test_loader


# ─── Train / eval ────────────────────────────────────────────────────────────

def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):
    model.train()
    loss_dis, loss_dis_tr = [], []
    loss_cla, loss_cla_tr = [], []
    loss_l1,  loss_rec, loss_rec_tr = [], [], []
    t0 = time.time()
    cer_tr, cer_te, cer_te2 = CER(), CER(), CER()

    for data in train_loader:
        rec_opt.zero_grad()
        l_rec_tr = model(data, epoch, 'rec_update', cer_tr)
        rec_opt.step()

        cla_opt.zero_grad()
        l_cla_tr = model(data, epoch, 'cla_update')
        cla_opt.step()

        dis_opt.zero_grad()
        l_dis_tr = model(data, epoch, 'dis_update')
        dis_opt.step()

        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_l1, l_rec = model(data, epoch, 'gen_update',
                                                     [cer_te, cer_te2])
        gen_opt.step()

        loss_dis.append(l_dis.item());    loss_dis_tr.append(l_dis_tr.item())
        loss_cla.append(l_cla.item());    loss_cla_tr.append(l_cla_tr.item())
        loss_l1.append(l_l1.item())
        loss_rec.append(l_rec.item());    loss_rec_tr.append(l_rec_tr.item())

    print('epo%d <tr>-<gen>: '
          'dis=%.2f-%.2f  cla=%.2f-%.2f  rec=%.2f-%.2f  '
          'l1=%.2f  cer=%.2f-%.2f-%.2f  t=%.1fs'
          % (epoch,
             np.mean(loss_dis_tr), np.mean(loss_dis),
             np.mean(loss_cla_tr), np.mean(loss_cla),
             np.mean(loss_rec_tr), np.mean(loss_rec),
             np.mean(loss_l1),
             cer_tr.fin(), cer_te.fin(), cer_te2.fin(),
             time.time() - t0))
    return cer_te.fin() + cer_te2.fin()


def _collect_images(loader, model, cer_obj1=None, cer_obj2=None, compute_loss=False):
    """
    Iterate loader once: collect ALL real images (every channel per writer)
    and regenerate matching fake images. Returns (accum, loss_lists).

    Using all NUM_CHANNEL images per writer gives ~93×50=4650 real samples
    for FID/KID — well above the 2048 minimum for a reliable covariance matrix.
    """
    accum = ImageAccumulator()
    loss_dis, loss_cla, loss_rec = [], [], []

    for data in loader:
        _, _, _, tr_img, _, _, img_xt, label_xt, label_xt_swap = data
        tr_img_gpu     = tr_img.to(gpu)
        label_xt_gpu   = label_xt.to(gpu)

        if compute_loss and cer_obj1 is not None:
            l_dis, l_cla, l_rec = model(data, 0, 'eval', [cer_obj1, cer_obj2])
            loss_dis.append(l_dis.item())
            loss_cla.append(l_cla.item())
            loss_rec.append(l_rec.item())

        with torch.no_grad():
            # Real: collect all NUM_CHANNEL images per writer (B, C, H, W) → (B*C, 1, H, W)
            B, C, H, W = tr_img_gpu.shape
            real_all_channels = tr_img_gpu.view(B * C, 1, H, W)

            # Fake: generate one image per writer, then repeat C times to match count
            f_xs              = model.gen.enc_image(tr_img_gpu)
            f_xt, f_embed     = model.gen.enc_text(label_xt_gpu, f_xs.shape)
            f_mix             = model.gen.mix(f_xs, f_embed)
            xg                = model.gen.decode(f_mix, f_xt)   # (B,1,H,W)
            fake_all_channels = xg.repeat(1, C, 1, 1).view(B * C, 1, H, W)

        accum.add(real_all_channels, fake_all_channels)

    return accum, loss_dis, loss_cla, loss_rec


def evaluate(test_loader, epoch, model_or_path,
             train_loader=None, logger=None,
             train_cer_gen=0.0, train_cer_swap=0.0):
    if isinstance(model_or_path, str):
        model = ConTranModel(NUM_WRITERS, SHOW_ITER_NUM, OOV).to(gpu)
        print('Loading', model_or_path)
        model.load_state_dict(torch.load(model_or_path))
    else:
        model = model_or_path
    model.eval()
    t0 = time.time()

    # ── Eval losses + CER on test writers ────────────────────────────────────
    cer_te, cer_te2 = CER(), CER()
    accum_test, loss_dis, loss_cla, loss_rec = _collect_images(
        test_loader, model, cer_te, cer_te2, compute_loss=True)

    dis_mean      = np.mean(loss_dis)
    cla_mean      = np.mean(loss_cla)
    rec_mean      = np.mean(loss_rec)
    eval_cer_gen  = cer_te.fin()
    eval_cer_swap = cer_te2.fin()

    print('EVAL: dis=%.3f  cla=%.3f  rec=%.3f  cer=%.2f-%.2f  t=%.1fs'
          % (dis_mean, cla_mean, rec_mean, eval_cer_gen, eval_cer_swap,
             time.time() - t0))

    # ── FID / KID over full dataset (train + test) ────────────────────────────
    # FID needs N >> 2048 for a valid covariance. Collecting all channels from
    # all 110 writers gives ~5500 images — enough for meaningful FID.
    metrics_result = {'fid': None, 'kid_mean': None, 'kid_std': None, 'n_samples': 0}
    try:
        # Collect train images too
        if train_loader is not None:
            accum_train, _, _, _ = _collect_images(train_loader, model)
            real_tr, fake_tr = accum_train.get()
        else:
            real_tr, fake_tr = None, None

        real_te, fake_te = accum_test.get()

        # Merge train + test
        parts_real = [x for x in [real_tr, real_te] if x is not None]
        parts_fake = [x for x in [fake_tr, fake_te] if x is not None]
        if parts_real and parts_fake:
            real_all = torch.cat(parts_real, dim=0)
            fake_all = torch.cat(parts_fake, dim=0)
            metrics_result = compute_metrics(real_all, fake_all, gpu)
    except Exception as e:
        print(f'  FID/KID computation failed: {e}')

    # ── Log to CSV ───────────────────────────────────────────────────────────
    if logger is not None:
        logger.log(epoch,
                   train_cer_gen, train_cer_swap,
                   eval_cer_gen,  eval_cer_swap,
                   metrics_result,
                   dis_mean, cla_mean, rec_mean)

    model.train()


# ─── Main ────────────────────────────────────────────────────────────────────

def main(train_loader, test_loader):
    model = ConTranModel(NUM_WRITERS, SHOW_ITER_NUM, OOV).to(gpu)

    if CurriculumModelID > 0:
        model_path = f'save_weights/contran-{CurriculumModelID}.model'
        print(f'Loading model: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=gpu))

    dis_opt = optim.Adam([p for p in model.dis.parameters() if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in model.gen.parameters() if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in model.rec.parameters() if p.requires_grad], lr=lr_rec)
    cla_opt = optim.Adam([p for p in model.cla.parameters() if p.requires_grad], lr=lr_cla)

    if CurriculumModelID > 0:
        opt_path = f'save_weights/contran-{CurriculumModelID}.optim'
        if os.path.exists(opt_path):
            print(f'Loading optimizer states: {opt_path}')
            opt_states = torch.load(opt_path, map_location='cpu')
            dis_opt.load_state_dict(opt_states['dis'])
            gen_opt.load_state_dict(opt_states['gen'])
            rec_opt.load_state_dict(opt_states['rec'])
            cla_opt.load_state_dict(opt_states['cla'])
        else:
            print(f'  [warn] no optimizer checkpoint at {opt_path} — '
                  f'Adam momentum will reset (expect ~50 epoch warm-up)')

    min_cer, min_idx, min_count = 1e9, 0, 0
    logger = MetricsLogger('metrics.csv')   # appends; safe to resume

    for epoch in range(CurriculumModelID, 50001):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)
        # cer is cer_te + cer_te2 from training gen_update pass
        train_cer_gen  = cer / 2.0   # approximate; exact values logged at eval
        train_cer_swap = cer / 2.0

        if epoch % MODEL_SAVE_EPOCH == 0:
            os.makedirs('save_weights', exist_ok=True)
            # Save model weights
            model_path = f'save_weights/contran-{epoch}.model'
            torch.save(model.state_dict(), model_path)
            # Save optimizer states (Adam momentum buffers) so resume is seamless
            opt_path = f'save_weights/contran-{epoch}.optim'
            torch.save({
                'dis': dis_opt.state_dict(),
                'gen': gen_opt.state_dict(),
                'rec': rec_opt.state_dict(),
                'cla': cla_opt.state_dict(),
                'epoch': epoch,
            }, opt_path)
            # Prune old checkpoints — keep only the last KEEP_LAST_N
            import glob as _glob
            all_models = sorted(_glob.glob('save_weights/contran-*.model'),
                                key=lambda p: int(p.split('-')[1].split('.')[0]))
            for old_ckpt in all_models[:-KEEP_LAST_N]:
                old_epoch = old_ckpt.split('-')[1].split('.')[0]
                os.remove(old_ckpt)
                old_opt = f'save_weights/contran-{old_epoch}.optim'
                if os.path.exists(old_opt):
                    os.remove(old_opt)

        if epoch % EVAL_EPOCH == 0:
            evaluate(test_loader, epoch, model,
                     train_loader=train_loader,
                     logger=logger,
                     train_cer_gen=train_cer_gen,
                     train_cer_swap=train_cer_swap)

        if EARLY_STOP_EPOCH is not None:
            if cer < min_cer:
                min_cer, min_idx, min_count = cer, epoch, 0
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print(f'Early stop at epoch {epoch}; best epoch {min_idx}')
                break


if __name__ == '__main__':
    print(time.ctime())

    # ── GPU sanity check ─────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Check driver and CUDA install.")
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"Free: {free_gb:.1f} GB / {total/1024**3:.1f} GB")
    if free_gb < 4.0:
        print(f"WARNING: Only {free_gb:.1f} GB free — another process may be "
              f"using the GPU.\n"
              f"Run:  nvidia-smi  to identify it, then kill it with:\n"
              f"      sudo kill -9 <PID>\n"
              f"or set CUDA_VISIBLE_DEVICES to a different GPU if available.")
    # ─────────────────────────────────────────────────────────────────────────

    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader)
    print(time.ctime())