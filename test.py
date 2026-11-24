# test.py
# Test script with:
# - Original HTML/image saving behavior
# - Metrics: L1, PSNR, SSIM, MS-SSIM, Balanced SCORE
# - Test metrics appended to checkpoints/<name>/log.txt
# - Fix for unaligned_mat slice indexing bug

import os
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from PIL import Image
import json
import re


def _safe_name(s: str) -> str:
    # make filesystem-safe folder name
    s = str(s)
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def _tensor_to_pil01(x: torch.Tensor) -> Image.Image:
    """
    x: tensor in [-1, 1], shape (C,H,W) or (1,C,H,W) or (H,W,C)
    returns PIL RGB image (uint8).
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()

    # handle batch
    if isinstance(x, torch.Tensor) and x.dim() == 4:
        x = x[0]

    # if CHW -> HWC
    if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    # if still tensor, clamp+scale
    if isinstance(x, torch.Tensor):
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)  # [0,1]
        x = (x * 255.0).round().to(torch.uint8).numpy()

    # now x is HWC uint8
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)

    return Image.fromarray(x, mode="RGB")


def save_recordwise_outputs(web_dir: str, img_path, visuals: dict, metrics_tuple):
    """
    Creates: <web_dir>/records/<record_name>/
      - real_A.png, real_B.png, fake_B.png
      - metrics.json (+ metrics.txt)
    """
    # record folder name based on path (already has your i suffix in some modes)
    base = _safe_name(os.path.splitext(os.path.basename(img_path[0]))[0])
    rec_dir = os.path.join(web_dir, "records", base)
    os.makedirs(rec_dir, exist_ok=True)

    # Save required images if present
    for k in ["real_A", "real_B", "fake_B"]:
        if k in visuals:
            try:
                im = _tensor_to_pil01(visuals[k])
                im.save(os.path.join(rec_dir, f"{k}.png"))
            except Exception as e:
                # Don't crash test for a save failure
                with open(os.path.join(rec_dir, "save_errors.txt"), "a") as f:
                    f.write(f"{k}: {repr(e)}\n")

    # Save metrics
    if metrics_tuple is not None:
        l1, psnr, ssim_v, ms_ssim_v = metrics_tuple
        payload = {
            "L1": float(l1),
            "PSNR": float(psnr),
            "SSIM": float(ssim_v),
            "MS_SSIM": float(ms_ssim_v),
        }
        with open(os.path.join(rec_dir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)

        with open(os.path.join(rec_dir, "metrics.txt"), "w") as f:
            f.write(f"L1:      {l1:.6f}\n")
            f.write(f"PSNR:    {psnr:.4f} dB\n")
            f.write(f"SSIM:    {ssim_v:.6f}\n")
            f.write(f"MS-SSIM: {ms_ssim_v:.6f}\n")


def calculate_psnr(fake_img, real_img, data_range=1.0, eps=1e-12):
    """
    Compute PSNR between two tensors.
    Assumes fake_img and real_img are normalized to [0, 1].
    Returns a python float.
    """
    mse = F.mse_loss(fake_img, real_img, reduction="mean").clamp_min(eps)
    # PSNR = 20*log10(MAX) - 10*log10(MSE)
    psnr = 20.0 * np.log10(float(data_range)) - 10.0 * torch.log10(mse).item()
    return float(psnr)


def balanced_score(psnr_avg, ssim_avg, ms_ssim_avg):
    """
    Score to maximize for early stopping / best checkpoint.
    """
    psnr_norm = psnr_avg / 50.0
    #return 0.50 * ms_ssim_avg + 0.30 * ssim_avg + 0.20 * psnr_norm
    return (psnr_norm + ssim_avg + ms_ssim_avg) / 3.0


@torch.no_grad()
def compute_metrics(fake_im, real_im):
    fake_im_norm = fake_im * 0.5 + 0.5
    real_im_norm = real_im * 0.5 + 0.5

    if real_im_norm.max() <= 0:
        return None

    l1_val = torch.abs(fake_im_norm - real_im_norm).mean().item()
    psnr_val = calculate_psnr(fake_im_norm, real_im_norm, data_range=1.0)
    ssim_val = ssim(fake_im_norm, real_im_norm, data_range=1.0).item()
    ms_ssim_val = ms_ssim(fake_im_norm, real_im_norm, data_range=1.0).item()
    return l1_val, psnr_val, ssim_val, ms_ssim_val


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # ----------------------------
    # HTML output (unchanged)
    # ----------------------------
    web_dir = os.path.join(
        opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}'
    )
    webpage = html.HTML(
        web_dir,
        f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}'
    )

    # ----------------------------
    # Metrics accumulators
    # ----------------------------
    total_l1 = total_psnr = total_ssim = total_ms_ssim = 0.0
    count = 0

    # ----------------------------
    # Test loop
    # ----------------------------
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()

        # ---- metrics ----
        fake_im = model.fake_B.cpu().data
        real_im = model.real_B.cpu().data
        m = compute_metrics(fake_im, real_im)
        if m is not None:
            l1, psnr, ssim_v, ms_ssim_v = m
            total_l1 += l1
            total_psnr += psnr
            total_ssim += ssim_v
            total_ms_ssim += ms_ssim_v
            count += 1

        # ---- visuals (original behavior + bugfix) ----
        if opt.dataset_mode == 'aligned_mat':
            visuals = model.get_current_visuals()
            visuals['real_A'] = visuals['real_A'][:, :, 0:3]
            visuals['real_B'] = visuals['real_B'][:, :, 0:3]
            visuals['fake_B'] = visuals['fake_B'][:, :, 0:3]
            img_path = model.get_image_paths()
            img_path[0] = img_path[0] + str(i)

        elif opt.dataset_mode == 'unaligned_mat':
            visuals = model.get_current_visuals()
            c = int(opt.input_nc) // 2  # FIXED
            slice_select = [c, c, c]
            visuals['real_A'] = visuals['real_A'][:, :, slice_select]
            visuals['real_B'] = visuals['real_B'][:, :, slice_select]
            visuals['fake_A'] = visuals['fake_A'][:, :, slice_select]
            visuals['fake_B'] = visuals['fake_B'][:, :, slice_select]
            visuals['rec_A'] = visuals['rec_A'][:, :, slice_select]
            visuals['rec_B'] = visuals['rec_B'][:, :, slice_select]
            img_path = model.get_image_paths()
            img_path[0] = img_path[0] + str(i)

        else:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

        print(f'{i:04d}: process image... {img_path}')
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
        # ---- record-wise save (images + metrics) ----
        save_recordwise_outputs(web_dir, img_path, visuals, m)

    webpage.save()

    # ----------------------------
    # Final metrics
    # ----------------------------
    if count > 0:
        l1_avg = total_l1 / count
        psnr_avg = total_psnr / count
        ssim_avg = total_ssim / count
        ms_ssim_avg = total_ms_ssim / count
    else:
        l1_avg = psnr_avg = ssim_avg = ms_ssim_avg = 0.0

    score = balanced_score(psnr_avg, ssim_avg, ms_ssim_avg)

    # ----------------------------
    # LOG TO CHECKPOINT DIRECTORY
    # ----------------------------
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'test.txt')

    with open(log_path, 'a') as logger:
        logger.write('\n')
        logger.write('==========================\n')
        logger.write(f'TEST RESULTS ({opt.which_epoch})\n')
        logger.write('==========================\n')
        logger.write(f'L1:        {l1_avg:.6f}\n')
        logger.write(f'PSNR:      {psnr_avg:.4f} dB\n')
        logger.write(f'SSIM:      {ssim_avg:.6f}\n')
        logger.write(f'MS-SSIM:   {ms_ssim_avg:.6f}\n')
        logger.write(f'SCORE:     {score:.6f}\n')
        logger.write(f'Samples:   {count}\n')
        logger.write('==========================\n\n')

    # Console summary
    print('\n==========================')
    print('TEST METRICS SUMMARY')
    print('Checkpoints log:', log_path)
    print('HTML results:', web_dir)
    print('==========================')
    print(f'L1:        {l1_avg:.6f}')
    print(f'PSNR:      {psnr_avg:.4f} dB')
    print(f'SSIM:      {ssim_avg:.6f}')
    print(f'MS-SSIM:   {ms_ssim_avg:.6f}')
    print(f'SCORE:     {score:.6f}')
    print('==========================\n')
