from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import matplotlib.pyplot as plt

#import wandb

#wandb.init(project="iilDepth-Testing")

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        # ------- Paths & split -------
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            f"Cannot find a folder at {opt.load_weights_folder}"
        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

               # ------- Dataset / Dataloader -------
        HEIGHT, WIDTH = opt.height, opt.width
        img_ext = '.png' if opt.png else '.jpg'
        dataset = datasets.SCAREDRAWDataset(
            opt.data_path, filenames, HEIGHT, WIDTH, [0], 4, is_train=False, img_ext=img_ext
        )
        dataloader = DataLoader(
            dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
            pin_memory=True, drop_last=False
        )

        # ---------- MODELOS (MPViT o ResNet) + CARGA ROBUSTA ----------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        def _unwrap_state_dict(ckpt, pref_candidates):
            """Devuelve un state_dict plano desde distintos formatos de guardado."""
            sd = ckpt
            if isinstance(sd, dict):
                for k in pref_candidates:
                    if k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                        break
            if isinstance(sd, dict):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            return sd if isinstance(sd, dict) else {}

        # ---- encoder ----
        if getattr(opt, "backbone", "resnet").lower() == "mpvit":
            print("[backbone] MPViT + auto-decoder")
            # canales de salida de MPViT por nivel: [64, 128, 216, 288, 288]
            encoder = networks.mpvit_small(use_pretrained=False)   # no imagenet
            encoder.num_ch_enc = [64, 128, 216, 288, 288]          # ← MUY IMPORTANTE
        else:
            print("[backbone] ResNet + DepthDecoder")
            encoder = networks.ResnetEncoder(opt.num_layers, False)

        ckpt_enc = torch.load(encoder_path, map_location="cpu")
        sd_enc = _unwrap_state_dict(ckpt_enc, ["model", "state_dict", "encoder", "encoder_state_dict", "net", "weights"])
        # si el sd viene con prefijo "encoder.", quítalo
        sd_enc = { (k[8:] if k.startswith("encoder.") else k): v for k, v in sd_enc.items() }
        # quita capas de clasificación si existieran
        sd_enc = { k: v for k, v in sd_enc.items() if not k.startswith("fc.") }

        missing_e, unexpected_e = encoder.load_state_dict(sd_enc, strict=False)
        print(f"[encoder] missing: {len(missing_e)} unexpected: {len(unexpected_e)}")

        # ---- decoder ----
        ckpt_dec = torch.load(decoder_path, map_location="cpu")
        sd_dec = _unwrap_state_dict(ckpt_dec, ["model", "state_dict", "decoder", "decoder_state_dict", "net", "weights"])

        # prueba primero con DepthDecoderT (transformer) y si no cuadra, usa DepthDecoder clásico
        decoder = None
        if hasattr(networks, "DepthDecoderT"):
            try:
                d = networks.DepthDecoderT()
                model_dict = d.state_dict()
                # filtra por nombre y shape
                filtered = {k: v for k, v in sd_dec.items() if k in model_dict and tuple(model_dict[k].shape) == tuple(v.shape)}
                if len(filtered) > 0:
                    decoder = d
                    missing_d, unexpected_d = decoder.load_state_dict(filtered, strict=False)
                    print(f"[decoder] DepthDecoderT | missing: {len(missing_d)} unexpected: {len(unexpected_d)}")
            except Exception:
                decoder = None

        if decoder is None:
            decoder = networks.DepthDecoder(encoder.num_ch_enc)
            missing_d, unexpected_d = decoder.load_state_dict(sd_dec, strict=False)
            print(f"[decoder] DepthDecoder | missing: {len(missing_d)} unexpected: {len(unexpected_d)}")

        encoder.to(device).eval()
        decoder.to(device).eval()
        # --------------------------------------------------------------




        pred_disps = []
        wmeta = getattr(opt, "width", WIDTH)
        hmeta = getattr(opt, "height", HEIGHT)
        print(f"-> Computing predictions with size {opt.width}x{opt.height}")


        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                outputs = decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)

                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        # --- Colapsar post-proc (Monodepth v1) de 2N -> N ---
        if opt.post_process:
            N = pred_disps.shape[0] // 2
            pred_disps = batch_post_process_disparity(
                pred_disps[:N],                   # disparidades originales
                pred_disps[N:, :, ::-1]           # disparidades de las imágenes volteadas (desvolteadas aquí)
            )


    else:
        # ------- Cargar predicciones desde archivo -------
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # ------- Ground truth -------
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, allow_pickle=True, encoding='latin1')["data"]

    print("-> Evaluating")
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors, ratios = [], []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-12)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth_m = gt_depth[mask]

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_m) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth_m, pred_depth))

    if not opt.disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    # ------- Guardar e imprimir resultados -------
    with open('results.txt', mode='a') as results_edit:
        results_edit.write("\n model_name: %s " % (opt.load_weights_folder))
        results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")



def colormap(inputs, normalize=True, torch_transpose=True):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        vis = inputs
        if normalize:
            ma = float(vis.max())
            mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / d

        if vis.ndim == 4:
            vis = vis.transpose([0, 2, 3, 1])
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, 0, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 3:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 2:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[..., :3]
            if torch_transpose:
                vis = vis.transpose(2, 0, 1)

        return vis

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())