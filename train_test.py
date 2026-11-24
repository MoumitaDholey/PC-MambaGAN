# train_test.py
# PC-MambaGAN: TRAIN + TEST in one script.
#
# Stage 1 - TRAIN
#   - trains on <dataroot>/train
#   - validates on <dataroot>/val every epoch (L1 / PSNR / SSIM / MS-SSIM)
#   - saves ONLY when the validation score improves -> 'latest' = best on validation
#   - early stopping (patience 20 epochs)
#
# Stage 2 - TEST
#   - reloads the 'latest' checkpoint (best on validation)
#   - tests every slice of <dataroot>/test, batch size 1, no flip, in order
#   - saves HTML page, record-wise images, metrics.json/txt and error maps
#   - writes a per-slice CSV with file name and patient ID -> <results>/<name>/test_latest/per_slice_metrics.csv
#   - appends the summary to <checkpoints>/<name>/test.txt
#
# Extra options (all others are the training options):
#   --results_dir   where test results go            (default ./results/)
#   --how_many      max test slices                  (default: all)
#   --aspect_ratio  aspect ratio of saved images     (default 1.0)
#   --skip_train    only run Stage 2 on an existing checkpoint
#   --fresh         start training from scratch even if an interrupted run exists
#   --patient_regex regex on the file name (without extension); group(1) = patient ID, group(2) = slice index
#
# RESTART / RESUME (automatic):
#   After every epoch the script saves  resume_net_G.pth, resume_net_D.pth, resume_optim.pth
#   and train_state.json in <checkpoints>/<name>/. If the job is killed, running the SAME command
#   again continues from the next epoch with the same weights, optimizer state, learning-rate
#   schedule, best validation score and early-stopping counter. If training had already
#   finished, it goes straight to testing.

import os
import re
import csv
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from PIL import Image

from models import create_model
from data import CreateDataLoader
from util.visualizer import Visualizer
from util import html
from options.train_options import TrainOptions


# ============================================================================
# Options: training options + test-stage options
# ============================================================================
class TrainTestOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves test results here')
        self.parser.add_argument('--how_many', type=int, default=1000000, help='max number of test images')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--skip_train', action='store_true', help='skip training, only test latest checkpoint')
        self.parser.add_argument('--fresh', action='store_true',
                                 help='ignore an interrupted run and start training from epoch 1 '
                                      '(default: resume automatically from the last completed epoch)')
        self.parser.add_argument('--patient_regex', type=str, default=r'^(.+?)[_-]+(\d+)$',
                                 help='regex on file stem; group(1) = patient ID, group(2) = slice index')


# ============================================================================
# Shared helpers
# ============================================================================
def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + "\n")
        logger.flush()


def calculate_psnr(fake_img, real_img, data_range=1.0, eps=1e-12):
    mse = F.mse_loss(fake_img, real_img, reduction="mean").clamp_min(eps)
    psnr = 20.0 * np.log10(float(data_range)) - 10.0 * torch.log10(mse).item()
    return float(psnr)


# ---- validation score (used for checkpoint selection) ----
def balanced_score_train(psnr_avg, ssim_avg, ms_ssim_avg):
    psnr_norm = psnr_avg / 50.0
    return 0.50 * ms_ssim_avg + 0.30 * ssim_avg + 0.20 * psnr_norm


# ---- test score (reported in test.txt) ----
def balanced_score_test(psnr_avg, ssim_avg, ms_ssim_avg):
    psnr_norm = psnr_avg / 50.0
    return (psnr_norm + ssim_avg + ms_ssim_avg) / 3.0


# ============================================================================
# Stage 1: training
# ============================================================================
@torch.no_grad()
def run_validation(model, dataset_val):
    total_l1 = total_psnr = total_ssim = total_ms_ssim = 0.0
    count = 0

    netG = getattr(model, "netG", None)
    if netG is not None:
        netG_was_training = netG.training
        netG.eval()
    else:
        netG_was_training = None

    for _, data_val in enumerate(dataset_val):
        model.set_input(data_val)
        model.test()

        fake_im = model.fake_B.detach().cpu()
        real_im = model.real_B.detach().cpu()
        fake_im_norm = fake_im * 0.5 + 0.5
        real_im_norm = real_im * 0.5 + 0.5

        if not (torch.isfinite(fake_im_norm).all() and torch.isfinite(real_im_norm).all()):
            continue
        bs = int(fake_im_norm.size(0))
        if bs <= 0:
            continue

        l1_val = torch.abs(fake_im_norm - real_im_norm).mean().item()
        ssim_val = ssim(fake_im_norm, real_im_norm, data_range=1.0).item()
        ms_ssim_val = ms_ssim(fake_im_norm, real_im_norm, data_range=1.0).item()
        psnr_val = calculate_psnr(fake_im_norm, real_im_norm, data_range=1.0)

        total_l1 += l1_val * bs
        total_psnr += psnr_val * bs
        total_ssim += ssim_val * bs
        total_ms_ssim += ms_ssim_val * bs
        count += bs

    if netG is not None and netG_was_training:
        netG.train()

    if count > 0:
        l1_avg, psnr_avg = total_l1 / count, total_psnr / count
        ssim_avg, ms_ssim_avg = total_ssim / count, total_ms_ssim / count
    else:
        l1_avg = psnr_avg = ssim_avg = ms_ssim_avg = 0.0

    return {
        "l1": float(l1_avg), "psnr": float(psnr_avg), "ssim": float(ssim_avg),
        "ms_ssim": float(ms_ssim_avg),
        "score": float(balanced_score_train(psnr_avg, ssim_avg, ms_ssim_avg)),
        "count": int(count),
    }


STATE_FILE = "train_state.json"


def _read_state(save_dir):
    path = os.path.join(save_dir, STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_state(save_dir, state):
    path = os.path.join(save_dir, STATE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)          # atomic: a crash never leaves a half-written state file


def _save_resume_point(model, save_dir, state):
    """Weights + optimizer state of the LAST completed epoch (separate from 'latest' = best)."""
    model.save("resume")
    optims = [o.state_dict() for o in getattr(model, "optimizers", [])]
    tmp = os.path.join(save_dir, "resume_optim.pth.tmp")
    torch.save({"optimizers": optims}, tmp)
    os.replace(tmp, os.path.join(save_dir, "resume_optim.pth"))
    _write_state(save_dir, state)


def train_stage(opt):
    # -------- data --------
    opt.phase = "train"
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f"Training loader length: {len(data_loader)}")

    opt.phase = "val"
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    print(f"Validation loader length: {len(data_loader_val)}")

    # -------- resume? --------
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    logger_path = os.path.join(save_dir, "log.txt")
    state = None if opt.fresh else _read_state(save_dir)

    if state and state.get("finished"):
        print(f"Training already finished (best val epoch {state['best_epoch']}, score {state['best_score']:.6f}). "
              f"Going to testing. Use --fresh to retrain.", flush=True)
        return state["best_epoch"], state["best_score"]

    resuming = bool(state) and os.path.isfile(os.path.join(save_dir, "resume_net_G.pth"))
    if resuming:
        opt.continue_train = True
        opt.which_epoch = "resume"
        opt.epoch_count = int(state["last_epoch"]) + 1
        with open(logger_path, "a") as logger:
            print_log(logger, f"\n===== RESUMING at epoch {opt.epoch_count} "
                              f"(best so far: epoch {state['best_epoch']}, score {state['best_score']:.6f}, "
                              f"bad_epochs {state['bad_epochs']}) =====\n")
    else:
        if state and not opt.fresh:
            print("[WARN] train_state.json found but resume weights are missing - starting fresh", flush=True)
        opt.continue_train = False
        opt.epoch_count = 1
        with open(logger_path, "w+") as logger:
            print_log(logger, opt.name)
            print_log(logger, "")

    # -------- model --------
    model = create_model(opt)          # continue_train=True -> loads resume_net_G/D.pth
    if resuming:
        optim_path = os.path.join(save_dir, "resume_optim.pth")
        if os.path.isfile(optim_path):
            saved = torch.load(optim_path, map_location="cpu")
            for o, sd in zip(getattr(model, "optimizers", []), saved["optimizers"]):
                o.load_state_dict(sd)
        else:
            print("[WARN] resume_optim.pth missing - optimizer restarts (weights are resumed)", flush=True)
    visualizer = Visualizer(opt)
    total_steps = int(state.get("total_steps", 0)) if resuming else 0

    validate_every = 1
    patience = 20
    min_delta = 1e-4
    best_score = float(state["best_score"]) if resuming else -float("inf")
    best_epoch = int(state["best_epoch"]) if resuming else 0
    bad_epochs = int(state["bad_epochs"]) if resuming else 0
    last_epoch = opt.epoch_count - 1

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        last_epoch = epoch
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        t_data = 0.0
        opt.phase = "train"

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            bs = int(data["A"].size(0)) if isinstance(data, dict) and ("A" in data) else int(opt.batchSize)

            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_steps += bs
            epoch_iter += bs

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                temp_visuals = model.get_current_visuals()
                visualizer.display_current_results(temp_visuals, epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / max(bs, 1)
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / len(data_loader), opt, errors)

            iter_data_time = time.time()

        print("End of epoch %d / %d \t Time Taken: %d sec"
              % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time), flush=True)

        if epoch % validate_every == 0:
            with open(logger_path, "a") as logger:
                opt.phase = "val"
                metrics = run_validation(model, dataset_val)
                print_log(logger,
                          "Epoch %3d   L1: %.5f   PSNR: %.2f dB   SSIM: %.4f   MS-SSIM: %.4f   SCORE: %.6f   (N=%d)"
                          % (epoch, metrics["l1"], metrics["psnr"], metrics["ssim"],
                             metrics["ms_ssim"], metrics["score"], metrics["count"]))

                if metrics["score"] > best_score + min_delta:
                    best_score = metrics["score"]
                    best_epoch = epoch
                    bad_epochs = 0
                    print_log(logger, f"New BEST score={best_score:.6f} -> saving ONLY as latest")
                    model.save("latest")
                else:
                    bad_epochs += 1
                    print_log(logger, f"No improvement -> bad_epochs={bad_epochs}/{patience}")
                print_log(logger, "")

            if bad_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best score={best_score:.6f}", flush=True)
                _write_state(save_dir, {"last_epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
                                        "bad_epochs": bad_epochs, "total_steps": total_steps,
                                        "finished": True, "reason": "early_stopping"})
                break

        model.update_learning_rate()

        # ---- resume point: everything needed to continue from the next epoch ----
        _save_resume_point(model, save_dir, {
            "last_epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
            "bad_epochs": bad_epochs, "total_steps": total_steps, "finished": False,
        })

    st = _read_state(save_dir) or {}
    if not st.get("finished"):
        _write_state(save_dir, {"last_epoch": last_epoch, "best_epoch": best_epoch, "best_score": best_score,
                                "bad_epochs": bad_epochs, "total_steps": total_steps,
                                "finished": True, "reason": "max_epochs"})
    with open(logger_path, "a") as logger:
        print_log(logger, f"TRAINING DONE. Best validation epoch={best_epoch}, score={best_score:.6f}, "
                          f"last epoch={last_epoch}")

    # free GPU memory before the test stage
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, best_score


# ============================================================================
# Stage 2: testing
# ============================================================================
def _safe_name(s):
    s = str(s).replace("\\", "/").split("/")[-1]
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200]


def _tensor_to_pil01(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if isinstance(x, torch.Tensor) and x.dim() == 4:
        x = x[0]
    if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if isinstance(x, torch.Tensor):
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8).numpy()
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return Image.fromarray(x, mode="RGB")


def save_error_map(rec_dir, visuals):
    if "fake_B" not in visuals or "real_B" not in visuals:
        return
    try:
        fake, real = visuals["fake_B"], visuals["real_B"]
        if torch.is_tensor(fake):
            fake = fake.detach().cpu()
        if torch.is_tensor(real):
            real = real.detach().cpu()
        if isinstance(fake, torch.Tensor) and fake.dim() == 4:
            fake = fake[0]
        if isinstance(real, torch.Tensor) and real.dim() == 4:
            real = real[0]
        if isinstance(fake, torch.Tensor) and fake.dim() == 3 and fake.shape[0] in (1, 3):
            fake = fake.permute(1, 2, 0)
        if isinstance(real, torch.Tensor) and real.dim() == 3 and real.shape[0] in (1, 3):
            real = real.permute(1, 2, 0)
        if isinstance(fake, torch.Tensor):
            fake = (fake * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
        if isinstance(real, torch.Tensor):
            real = (real * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
        if fake.ndim == 2:
            fake = np.expand_dims(fake, axis=-1)
        if real.ndim == 2:
            real = np.expand_dims(real, axis=-1)

        err = np.abs(fake.astype(np.float32) - real.astype(np.float32))
        if err.ndim == 3 and err.shape[2] > 1:
            err = err.mean(axis=2)
        err_min, err_max = err.min(), err.max()
        err_norm = (err - err_min) / (err_max - err_min) if err_max > err_min else np.zeros_like(err, np.float32)

        x = err_norm
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        heatmap = (np.stack([r, g, b], axis=-1) * 255.0).round().astype(np.uint8)
        Image.fromarray(heatmap, mode="RGB").save(os.path.join(rec_dir, "error_map.png"))
    except Exception as e:
        with open(os.path.join(rec_dir, "save_errors.txt"), "a") as f:
            f.write(f"error_map: {repr(e)}\n")


def save_recordwise_outputs(web_dir, img_path, visuals, metrics_tuple):
    base = _safe_name(os.path.splitext(os.path.basename(img_path[0]))[0])
    rec_dir = os.path.join(web_dir, "records", base)
    os.makedirs(rec_dir, exist_ok=True)

    for k in ["real_A", "real_B", "fake_B"]:
        if k in visuals:
            try:
                _tensor_to_pil01(visuals[k]).save(os.path.join(rec_dir, f"{k}.png"))
            except Exception as e:
                with open(os.path.join(rec_dir, "save_errors.txt"), "a") as f:
                    f.write(f"{k}: {repr(e)}\n")

    if metrics_tuple is not None:
        l1, psnr, ssim_v, ms_ssim_v = metrics_tuple
        with open(os.path.join(rec_dir, "metrics.json"), "w") as f:
            json.dump({"L1": float(l1), "PSNR": float(psnr), "SSIM": float(ssim_v),
                       "MS_SSIM": float(ms_ssim_v)}, f, indent=2)
        with open(os.path.join(rec_dir, "metrics.txt"), "w") as f:
            f.write(f"L1:      {l1:.6f}\nPSNR:    {psnr:.4f} dB\nSSIM:    {ssim_v:.6f}\nMS-SSIM: {ms_ssim_v:.6f}\n")

    save_error_map(rec_dir, visuals)


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


def patient_and_slice(path, regex):
    """Return (patient_id, slice_index or None) from the file name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(regex, stem)
    if not m:
        return stem, None
    pid = m.group(1)
    sl = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2).isdigit() else None
    return pid, sl


def test_stage(opt, best_epoch=None, best_score=None):
    # switch options to test mode
    opt.isTrain = False
    opt.phase = "test"
    opt.which_epoch = "latest"
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.continue_train = False

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)            # isTrain=False -> loads latest_net_G.pth
    visualizer = Visualizer(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.which_epoch}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}")

    total_l1 = total_psnr = total_ssim = total_ms_ssim = 0.0
    count = 0
    rows = []

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()

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

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        src_path = img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path)
        pid, sl = patient_and_slice(src_path, opt.patient_regex)
        rows.append({
            "index": i,
            "file": os.path.basename(src_path),
            "patient_id": pid,
            "slice": sl if sl is not None else "",
            "L1": m[0] if m else "",
            "PSNR": m[1] if m else "",
            "SSIM": m[2] if m else "",
            "MS_SSIM": m[3] if m else "",
            "excluded_empty_target": 0 if m else 1,
        })

        print(f"{i:04d}: process image... {img_path}", flush=True)
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
        save_recordwise_outputs(web_dir, img_path, visuals, m)

    webpage.save()

    # ---- per-slice CSV ----
    csv_path = os.path.join(web_dir, "per_slice_metrics.csv")
    fields = ["index", "file", "patient_id", "slice", "L1", "PSNR", "SSIM", "MS_SSIM", "excluded_empty_target"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    if count > 0:
        l1_avg, psnr_avg = total_l1 / count, total_psnr / count
        ssim_avg, ms_ssim_avg = total_ssim / count, total_ms_ssim / count
    else:
        l1_avg = psnr_avg = ssim_avg = ms_ssim_avg = 0.0
    score = balanced_score_test(psnr_avg, ssim_avg, ms_ssim_avg)
    n_patients = len({r["patient_id"] for r in rows})

    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "test.txt")
    lines = [
        "", "==========================",
        f"TEST RESULTS ({opt.which_epoch})"
        + (f"  [best val epoch {best_epoch}, val score {best_score:.6f}]" if best_epoch else ""),
        "==========================",
        f"L1:        {l1_avg:.6f}",
        f"PSNR:      {psnr_avg:.4f} dB",
        f"SSIM:      {ssim_avg:.6f}",
        f"MS-SSIM:   {ms_ssim_avg:.6f}",
        f"SCORE:     {score:.6f}",
        f"Samples:   {count}  (excluded empty targets: {len(rows) - count})",
        f"Patients:  {n_patients}   (IDs from file name)",
        f"Per-slice: {csv_path}",
        "==========================", "",
    ]
    with open(log_path, "a") as logger:
        logger.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("HTML results:", web_dir, flush=True)


# ============================================================================
if __name__ == "__main__":
    opt = TrainTestOptions().parse()

    best_epoch, best_score = None, None
    if not opt.skip_train:
        best_epoch, best_score = train_stage(opt)
    else:
        print("--skip_train set: testing existing 'latest' checkpoint", flush=True)

    test_stage(opt, best_epoch, best_score)
