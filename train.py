# train_clean.py (or traintest.py but training-only)
# Clean TRAIN-ONLY script with:
# - Validation metrics (L1/PSNR/SSIM/MS-SSIM) every epoch
# - Balanced-score early stopping
# - Save ONLY ONE checkpoint: latest (best on validation score)
# - NO test phase execution
#
# ADDED:
# - Table-IV style profiling for our generator:
#   Params (M), Inference time (ms), Peak GPU memory (GB)

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim

from models import create_model
from data import CreateDataLoader
from util.visualizer import Visualizer
from options.train_options import TrainOptions


def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')


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
    # return 0.50 * ms_ssim_avg + 0.30 * ssim_avg + 0.20 * psnr_norm
    return (psnr_norm + ssim_avg + ms_ssim_avg) / 3.0


# -------------------------
# ADDED: Table-IV helpers
# -------------------------
def get_generator_module(model):
    # common wrappers
    for attr in ["netG", "netG_A", "netG_B", "G", "generator"]:
        if hasattr(model, attr):
            net = getattr(model, attr)
            if net is not None:
                return net
    raise AttributeError("Could not find generator in model (netG/netG_A/netG_B/G/generator).")


def count_parameters_million_total(net):
    # TOTAL params (Table IV)
    return sum(p.numel() for p in net.parameters()) / 1e6


@torch.no_grad()
def benchmark_generator_table_iv(model, dataset_val, warmup=20, iters=100):
    """
    Returns:
      avg_inference_ms, peak_mem_gb

    Measures generator forward via model.test() (generator-only forward path).
    Uses CUDA events for accurate timing.
    """
    use_cuda = torch.cuda.is_available()

    # take one batch
    batch = next(iter(dataset_val))

    # ensure eval-ish forward (your test() should do no-grad forward)
    # Warmup (GPU kernels, caching)
    for _ in range(warmup):
        model.set_input(batch)
        model.test()

    if not use_cuda:
        # CPU fallback (not comparable to Table IV GPU mem)
        t0 = time.time()
        for _ in range(iters):
            model.set_input(batch)
            model.test()
        t1 = time.time()
        avg_ms = (t1 - t0) / iters * 1000.0
        return avg_ms, 0.0

    # GPU: accurate timing + peak memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        model.set_input(batch)
        model.test()
    end_event.record()

    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)  # milliseconds over iters
    avg_ms = total_ms / iters

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return avg_ms, peak_mem_gb



@torch.no_grad()
def run_validation(model, dataset_val):
    """
    Compute validation metrics (fake_B vs real_B), assumes outputs in [-1,1].
    Returns averages and balanced score.
    """
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_ms_ssim = 0.0
    count = 0

    for _, data_val in enumerate(dataset_val):
        model.set_input(data_val)
        model.test()

        fake_im = model.fake_B.cpu().data
        real_im = model.real_B.cpu().data

        # Normalize to [0,1]
        fake_im_norm = fake_im * 0.5 + 0.5
        real_im_norm = real_im * 0.5 + 0.5

        if real_im_norm.max() <= 0:
            continue

        l1_val = torch.abs(fake_im_norm - real_im_norm).mean().item()
        psnr_val = calculate_psnr(fake_im_norm, real_im_norm, data_range=1.0)
        ssim_val = ssim(fake_im_norm, real_im_norm, data_range=1.0).item()
        ms_ssim_val = ms_ssim(fake_im_norm, real_im_norm, data_range=1.0).item()

        total_l1 += l1_val
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_ms_ssim += ms_ssim_val
        count += 1

    if count > 0:
        l1_avg = total_l1 / count
        psnr_avg = total_psnr / count
        ssim_avg = total_ssim / count
        ms_ssim_avg = total_ms_ssim / count
    else:
        l1_avg = psnr_avg = ssim_avg = ms_ssim_avg = 0.0

    score = balanced_score(psnr_avg, ssim_avg, ms_ssim_avg)

    return {
        "l1": l1_avg,
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "ms_ssim": ms_ssim_avg,
        "score": score,
        "count": count,
    }


if __name__ == '__main__':
    opt = TrainOptions().parse()

    # -------------------------
    # Load train data
    # -------------------------
    opt.phase = 'train'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f'Training set size: {len(data_loader)}')

    # -------------------------
    # Load val data
    # -------------------------
    opt.phase = 'val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    print(f'Validation set size: {len(data_loader_val)}')

    # -------------------------
    # Logger
    # -------------------------
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    logger_path = os.path.join(save_dir, 'log.txt')

    with open(logger_path, 'w+') as logger:
        print_log(logger, opt.name)
        print_log(logger, '')

    # Keep this dead code to match your original structure (harmless)
    if opt.model == 'cycle_gan':
        _ = np.zeros([2, opt.niter + opt.niter_decay, len(dataset_val)])
    else:
        _ = np.zeros([opt.niter + opt.niter_decay, len(dataset_val)])

    # -------------------------
    # Model
    # -------------------------
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # -------------------------
    # ADDED: Table-IV style profiling (once, right after model init)
    # -------------------------

    try:
        netG = get_generator_module(model)
        param_m = count_parameters_million_total(netG)

        opt.phase = "val"
        inf_ms, mem_gb = benchmark_generator_table_iv(model, dataset_val, warmup=20, iters=100)

        with open(logger_path, "a") as logger:
            print_log(logger, "=== Model Complexity (Table-IV style) ===")
            print_log(logger, f"Params (M): {param_m:.2f}")
            print_log(logger, f"Inference (ms): {inf_ms:.2f}")
            print_log(logger, f"Memory (GB): {mem_gb:.2f}")
            print_log(logger, "=======================================")
            print_log(logger, "")

        print(f"[Table-IV] Params(M)={param_m:.2f}, Inf(ms)={inf_ms:.2f}, Mem(GB)={mem_gb:.2f}")

    except Exception as e:
        with open(logger_path, "a") as logger:
            print_log(logger, f"[WARN] Could not compute Table-IV stats: {repr(e)}")
            print_log(logger, "")
        print(f"[WARN] Could not compute Table-IV stats: {repr(e)}")


    # -------------------------
    # Dynamic training controls
    # -------------------------
    validate_every = 1      # validate every epoch
    patience = 10           # early stop after N non-improving validations
    min_delta = 1e-4        # min improvement in score

    best_score = -float('inf')
    bad_epochs = 0

    # IMPORTANT: only save 'latest' when improved; disable step-wise saving
    save_latest_during_steps = False

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        opt.phase = 'train'

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0

                if opt.dataset_mode == 'aligned_mat':
                    temp_visuals = model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif opt.dataset_mode == 'unaligned_mat':
                    temp_visuals = model.get_current_visuals()
                    temp_visuals['real_A'] = temp_visuals['real_A'][:, :, 0:3]
                    temp_visuals['real_B'] = temp_visuals['real_B'][:, :, 0:3]
                    temp_visuals['fake_A'] = temp_visuals['fake_A'][:, :, 0:3]
                    temp_visuals['fake_B'] = temp_visuals['fake_B'][:, :, 0:3]
                    temp_visuals['rec_A'] = temp_visuals['rec_A'][:, :, 0:3]
                    temp_visuals['rec_B'] = temp_visuals['rec_B'][:, :, 0:3]

                    if opt.lambda_identity > 0:
                        temp_visuals['idt_A'] = temp_visuals['idt_A'][:, :, 0:3]
                        temp_visuals['idt_B'] = temp_visuals['idt_B'][:, :, 0:3]

                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                else:
                    temp_visuals = model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / len(data_loader), opt, errors)

            # Disable step-wise save by default (prevents overwriting best)
            if save_latest_during_steps and (total_steps % opt.save_latest_freq == 0):
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # -------------------------
        # Validation + early stopping
        # -------------------------
        if epoch % validate_every == 0:
            with open(logger_path, 'a') as logger:
                opt.phase = 'val'
                metrics = run_validation(model, dataset_val)

                print_log(logger,
                          'Epoch %3d   L1: %.5f   PSNR: %.2f dB   SSIM: %.4f   MS-SSIM: %.4f   SCORE: %.6f   (N=%d)' %
                          (epoch, metrics['l1'], metrics['psnr'], metrics['ssim'], metrics['ms_ssim'],
                           metrics['score'], metrics['count']))

                # Save ONLY when improved (best -> latest)
                if metrics['score'] > best_score + min_delta:
                    best_score = metrics['score']
                    bad_epochs = 0
                    print_log(logger, f'New BEST score={best_score:.6f} -> saving ONLY as latest')
                    model.save('latest')
                else:
                    bad_epochs += 1
                    print_log(logger, f'No improvement -> bad_epochs={bad_epochs}/{patience}')

                print_log(logger, '')

            if bad_epochs >= patience:
                print(f'Early stopping triggered at epoch {epoch}. Best score={best_score:.6f}')
                break

        # Update learning rate
        model.update_learning_rate()
