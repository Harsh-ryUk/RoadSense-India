"""Train a binary SegFormer on IDD Lite; no official-val inference during training."""
import argparse
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_idd_lite_training import digest, load_frozen
from src.evaluation.ground_truth import binary_road_counts, road_summary


def remap_labels(raw):
    if raw.ndim != 2 or not set(np.unique(raw)) <= set(range(7)) | {255}:
        raise ValueError('Expected official IDD Lite 2D level1Id mask')
    labels = (raw == 0).astype(np.int64)
    labels[raw == 255] = 255
    return labels


class DrivableDataset(Dataset):
    def __init__(self, root, rows, processor, size, augment=False):
        self.root, self.rows, self.size, self.augment = Path(root), rows, size, augment
        self.mean = np.array(processor.image_mean, np.float32).reshape(1, 1, 3)
        self.std = np.array(processor.image_std, np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        bgr = cv2.imread(str(self.root / row['image']))
        raw = cv2.imread(str(self.root / row['mask']), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.shape != bgr.shape[:2]:
            raise ValueError('Unreadable or mismatched frozen image/mask')
        rgb, labels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), remap_labels(raw)
        if self.augment:
            if random.random() < 0.5:
                rgb, labels = rgb[:, ::-1], labels[:, ::-1]
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.8, 1.2)
                          + random.uniform(-15, 15), 0, 255).astype(np.uint8)
        rgb = np.array(Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR))
        values = (rgb.astype(np.float32) / 255 - self.mean) / self.std
        return torch.from_numpy(values.transpose(2, 0, 1).copy()), torch.from_numpy(labels.copy())


def segmentation_loss(logits, labels):
    # Resize and loss reductions in FP32, even when the model forward uses AMP.
    logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode='bilinear', align_corners=False)
    valid = labels != 255
    if not torch.any(valid):
        raise ValueError('Batch contains no valid labels')
    ce = F.cross_entropy(logits, labels, ignore_index=255)
    road = logits.softmax(1)[:, 1][valid]
    truth = (labels[valid] == 1).float()
    dice = 1 - (2 * (road * truth).sum() + 1) / (road.sum() + truth.sum() + 1)
    return ce + 0.5 * dice


def safe_optimizer_step(parameters, optimizer, scaler, scheduler):
    """Never apply nonfinite gradients; allow GradScaler to recover AMP overflow."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        raise ValueError('No gradients produced')
    scaler.unscale_(optimizer)
    finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).all().item()
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError('Nonfinite gradients without AMP; refusing unsafe optimizer update')
        scale_before = scaler.get_scale()
        # unscale_ has already recorded the overflow: step skips optimizer.step.
        scaler.step(optimizer)
        scaler.update()
        if not scaler.get_scale() < scale_before:
            raise RuntimeError('AMP overflow did not reduce the loss scale')
        optimizer.zero_grad(set_to_none=True)
        return False
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return True


def validate_raw(model, loader, device):
    model.eval()
    counts = []
    with torch.inference_mode():
        for values, labels in loader:
            logits = model(pixel_values=values.to(device)).logits
            prediction = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear',
                                       align_corners=False).argmax(1).cpu().numpy()
            for mask, truth in zip(prediction, labels.numpy()):
                counts.append(binary_road_counts(mask, truth, [1], [0, 1], [255]))
    return road_summary(counts)


def target_status(metrics):
    return {'drivable_IoU_95': metrics['IoU'] is not None and metrics['IoU'] >= 0.95,
            'precision_98': metrics['precision'] is not None and metrics['precision'] >= 0.98,
            'recall_95': metrics['recall'] is not None and metrics['recall'] >= 0.95}


def train(args):
    if args.epochs < 1 or args.batch < 1 or args.input_size < 32 or args.threads < 1 or args.smoke_steps < 0 or args.lr <= 0:
        raise ValueError('Invalid training settings')
    precision = getattr(args, 'precision', 'fp16')
    if precision not in ('fp16', 'fp32'):
        raise ValueError('Unsupported training precision')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested training device unavailable; refusing silent CPU fallback')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; existing experiments are immutable')
    manifest = load_frozen(args.manifest, args.root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    processor = SegformerImageProcessor.from_pretrained(args.model, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model, num_labels=2,
        id2label={0: 'non_drivable', 1: 'drivable'}, label2id={'non_drivable': 0, 'drivable': 1},
        ignore_mismatched_sizes=True).to(device)
    fit = DrivableDataset(args.root, manifest['splits']['fit'], processor, args.input_size, augment=True)
    dev = DrivableDataset(args.root, manifest['splits']['development'], processor, args.input_size)
    # Fixed original label sizes, not resized labels, are used for both loss and metrics.
    if len({tuple(row['resolution']) for rows in manifest['splits'].values() for row in rows}) != 1:
        raise ValueError('This batched trainer requires uniform official IDD Lite source dimensions')
    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(fit, batch_size=args.batch, shuffle=True, generator=generator, num_workers=0)
    dev_loader = DataLoader(dev, batch_size=args.batch, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(fit_loader)
    if args.smoke_steps > total_steps:
        raise ValueError('smoke_steps exceeds the available optimizer steps')
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_amp = device.type == 'cuda' and precision == 'fp16'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'schema_version': 1, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'settings': vars(args), 'manifest_sha256': digest(args.manifest),
                'counts': manifest['counts'], 'model_revision': getattr(model.config, '_commit_hash', None),
                'label_schema': manifest['label_schema'], 'loss': 'ignore-aware CE + 0.5 soft drivable Dice',
                'selection_metric': 'Prefer all three targets met on internal raw development metrics; otherwise maximize raw global IoU. No official-val selection.',
                'official_val_used_for_selection': False, 'official_val_inference_during_training': False,
                'numerics': {'precision': 'cuda_fp16_amp' if use_amp else 'fp32',
                             'loss_and_resize_dtype': 'float32',
                             'amp_overflow_policy': 'Skip unsafe update, reduce scale, do not advance scheduler; fail after 8 consecutive overflows'},
                'environment': {'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                'python': platform.python_version(), 'torch': torch.__version__,
                                'packages': {name: importlib.metadata.version(name) for name in
                                             ('transformers', 'numpy', 'opencv-python', 'Pillow')}},
                'source_sha256': {str(Path(__file__).relative_to(ROOT)): digest(__file__),
                                  'scripts/prepare_idd_lite_training.py': digest(ROOT / 'scripts/prepare_idd_lite_training.py'),
                                  'src/evaluation/ground_truth.py': digest(ROOT / 'src/evaluation/ground_truth.py')},
                'limitations': ['Seeds do not guarantee bit-identical GPU training.',
                                'Internal raw-model metrics differ from deployed mask-postprocessed accuracy.',
                                'User targets are ambitions, not achieved accuracy.']}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    best_iou, best_rank, history, completed_steps = -1, (False, -1), [], 0
    skipped_steps, consecutive_overflows = 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, weighted_loss, seen = time.perf_counter(), 0.0, 0
        epoch_updates, epoch_skipped = 0, 0
        for batch_index, (values, labels) in enumerate(fit_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pixel_values=values.to(device)).logits
            with torch.autocast(device_type=device.type, enabled=False):
                loss = segmentation_loss(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            if not safe_optimizer_step(model.parameters(), optimizer, scaler, scheduler):
                skipped_steps += 1
                epoch_skipped += 1
                consecutive_overflows += 1
                event = {'event': 'amp_overflow_skipped', 'epoch': epoch, 'batch': batch_index,
                         'loss_scale': scaler.get_scale(), 'skipped_steps_total': skipped_steps,
                         'consecutive_overflows': consecutive_overflows}
                with (output / 'numerical_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
                if consecutive_overflows >= 8:
                    raise RuntimeError('8 consecutive AMP overflows; stop and investigate or run a named FP32 experiment')
                continue
            consecutive_overflows = 0
            completed_steps += 1
            epoch_updates += 1
            seen += len(labels)
            weighted_loss += loss.item() * len(labels)
            if args.smoke_steps and completed_steps >= args.smoke_steps:
                smoke = {'purpose': 'finite-loss/backpropagation smoke test, NOT accuracy or completed fine-tuning',
                         'steps': completed_steps, 'mean_loss': weighted_loss / seen,
                         'elapsed_seconds': time.perf_counter() - start, 'accuracy': None}
                (output / 'smoke.json').write_text(json.dumps(smoke, indent=2))
                print(json.dumps(smoke, indent=2), flush=True)
                return smoke
        if not seen:
            raise RuntimeError('Epoch produced no successful optimizer updates')
        metrics = validate_raw(model, dev_loader, device)
        row = {'epoch': epoch, 'fit_mean_loss': weighted_loss / seen,
               'optimizer_updates': epoch_updates, 'amp_skipped_updates': epoch_skipped,
               'optimizer_updates_total': completed_steps, 'amp_skipped_updates_total': skipped_steps,
               'loss_scale': scaler.get_scale(),
               'epoch_seconds_including_dev': time.perf_counter() - start,
               'development_raw': metrics, 'development_raw_target_status': target_status(metrics)}
        history.append(row)
        rank = (all(target_status(metrics).values()), metrics['IoU'] if metrics['IoU'] is not None else -1)
        if rank > best_rank:
            best_rank = rank
            best_iou = metrics['IoU']
            model.save_pretrained(output / 'best')
            processor.save_pretrained(output / 'best')
            (output / 'best_selection.json').write_text(json.dumps(row, indent=2))
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    config = {'system': {'device': str(device)}, 'zero_shot': {'enabled': False},
              'segmentation': {'model_name': str(output / 'best'), 'input_size': [args.input_size, args.input_size],
                               'road_class_ids': [1], 'frame_roi': [0, 0, 1, 1],
                               'roi_top_fraction': 0, 'min_road_coverage': 0.02}}
    (output / 'evaluation_config.yaml').write_text(yaml.safe_dump(config))
    weights = output / 'best/model.safetensors'
    if not weights.is_file():
        raise ValueError('Missing selected checkpoint')
    summary = {'training_completed': True, 'best_internal_raw_IoU': best_iou,
               'completed_epochs': len(history), 'optimizer_updates_total': completed_steps,
               'amp_skipped_updates_total': skipped_steps,
               'best_weights_sha256': digest(weights), 'official_val_metrics': None,
               'instruction': 'Freeze the selected model/config, then run evaluate_accuracy.py separately. Never claim raw internal-dev scores as official-val or deployed accuracy.'}
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='nvidia/segformer-b0-finetuned-ade-512-512')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16',
                        help='FP16 AMP on CUDA (overflow-safe); FP32 disables mixed precision. CPU always uses FP32.')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
