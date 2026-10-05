"""Evaluate a frozen, annotated held-out manifest. No labels means no accuracy.

python scripts/evaluate_accuracy.py --manifest runs/idd_val_manifest.json --config configs/idd_eval.yaml --device cuda
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_pipeline import benchmark_device, digest
from src.evaluation.ground_truth import (
    binary_road_counts,
    coco_detection_metrics,
    road_summary,
)


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('split') not in ('val', 'test'):
        raise ValueError('Manifest must use schema_version=1 and a held-out val/test split')
    samples = manifest.get('samples', [])
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Manifest must contain nonempty, uniquely identified samples')
    categories = manifest.get('categories', [])
    if len(set(categories)) != len(categories):
        raise ValueError('Category names must be unique')
    for sample in samples:
        if 'mask' not in sample and 'boxes' not in sample:
            raise ValueError('Every sample needs a mask or an explicitly annotated boxes list')
        for key in ('image', 'mask'):
            if key in sample:
                resolved = (path.parent / sample[key]).resolve()
                if not resolved.is_file():
                    raise ValueError(f'Missing {key}: {resolved}')
                if sample.get(key + '_sha256') and digest(resolved) != sample[key + '_sha256']:
                    raise ValueError(f'Changed frozen {key}: {resolved}')
                sample[key + '_path'] = resolved
        for box in sample.get('boxes', []):
            if box['category'] not in categories:
                raise ValueError(f'Undeclared box category: {box["category"]}')
    return manifest


def local_segformer_fingerprint(model_name):
    directory = Path(model_name)
    if not directory.is_dir():
        return None  # Hub models are identified by their resolved checkpoint revision.
    paths = {directory / 'config.json', directory / 'preprocessor_config.json',
             *directory.glob('*.safetensors'), *directory.glob('pytorch_model*.bin'),
             *directory.glob('*.index.json')}
    return {path.name: digest(path) for path in sorted(paths) if path.is_file()}


def check_evaluation_contract(manifest, config_path, model_name):
    """An independent test must use the checkpoint/config frozen before prediction."""
    contract = manifest.get('evaluation_contract')
    if contract is None:
        return  # Existing published IDD protocols remain unchanged.
    if manifest.get('stage') != 'frozen_human_reviewed_test':
        raise ValueError('Independent test is not frozen and human reviewed')
    if digest(config_path) != contract.get('configuration_file_sha256'):
        raise ValueError('Independent-test evaluation configuration changed')
    fingerprint = local_segformer_fingerprint(model_name)
    if not fingerprint or fingerprint != contract.get('segformer_local_files_sha256'):
        raise ValueError('Independent-test checkpoint changed')


def evaluate(args):
    import torch

    from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
    from src.perception.india_detector import IndiaObjectDetector
    from src.utils.runtime import validate_config

    manifest = load_manifest(args.manifest)
    config = validate_config(yaml.safe_load(Path(args.config).read_text()))
    device = benchmark_device(args.device)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    has_masks = any('mask' in sample for sample in manifest['samples'])
    has_boxes = any('boxes' in sample for sample in manifest['samples'])
    seg = config.get('segmentation', {})
    segformer_name = seg.get('model_name', 'nvidia/segformer-b0-finetuned-ade-512-512')
    check_evaluation_contract(manifest, args.config, segformer_name)
    local_checkpoint = local_segformer_fingerprint(segformer_name) if has_masks else None
    lane = SegFormerLaneDetector(device=device, **{key: value for key, value in seg.items() if key in
        ('model_name', 'input_size', 'road_class_ids', 'frame_roi', 'roi_top_fraction', 'min_road_coverage')}) if has_masks else None
    det = config.get('detection', {})
    weights = Path(det.get('model_path', 'yolov8n.pt'))
    if not weights.is_absolute() and (ROOT / weights).is_file():
        weights = ROOT / weights
    detector = IndiaObjectDetector(model_path=str(weights), device=device,
        conf_thres=det.get('confidence_threshold', 0.35), iou_thres=det.get('iou_threshold', 0.45),
        input_size=det.get('input_size', [640, 640]), category_thresholds=det.get('category_thresholds')) if has_boxes else None
    category_names = manifest.get('categories', [])
    category_ids = {name: index + 1 for index, name in enumerate(category_names)}
    images, annotations, predictions, mask_counts, trace, gt_counts = [], [], [], [], [], Counter()
    for index, sample in enumerate(manifest['samples'], start=1):
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            raise ValueError(f'Cannot decode image {sample["id"]}')
        h, w = image.shape[:2]
        row = {'id': sample['id'], 'image_sha256': digest(sample['image_path']), 'resolution': [w, h]}
        if 'mask' in sample:
            labels = cv2.imread(str(sample['mask_path']), cv2.IMREAD_UNCHANGED)
            if labels is None:
                raise ValueError(f'Cannot decode mask {sample["id"]}')
            result = lane.detect(image)
            schema = manifest['label_schema']
            counts = binary_road_counts(result['lane_mask'], labels, schema['road_ids'], schema['valid_ids'], schema['ignore_ids'])
            mask_counts.append(counts)
            row.update(mask_sha256=digest(sample['mask_path']), road_status=result['road_status'], **counts)
        if 'boxes' in sample:
            images.append({'id': index, 'width': w, 'height': h})
            for box in sample['boxes']:
                x1, y1, x2, y2 = box['bbox_xyxy']
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f'Invalid pixel box in {sample["id"]}')
                annotations.append({'id': len(annotations) + 1, 'image_id': index,
                    'category_id': category_ids[box['category']], 'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1), 'iscrowd': int(box.get('iscrowd', 0))})
                gt_counts[box['category']] += 1
            for prediction in detector.detect(image).detections:
                if prediction.class_name not in category_ids:
                    continue  # Outside the explicitly declared evaluation taxonomy.
                x1, y1, x2, y2 = prediction.bbox
                predictions.append({'image_id': index, 'category_id': category_ids[prediction.class_name],
                                    'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': prediction.confidence})
        trace.append(row)
        if index % 20 == 0:
            print(f'Evaluated {index}/{len(manifest["samples"])} held-out images', flush=True)
    if has_masks and local_segformer_fingerprint(segformer_name) != local_checkpoint:
        raise ValueError('Local checkpoint files changed during evaluation')
    check_evaluation_contract(manifest, args.config, segformer_name)
    report = {
        'schema_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': manifest['dataset'], 'split': manifest['split'], 'images': len(trace),
        'selection': manifest.get('selection'),
        'evaluation_contract': manifest.get('evaluation_contract'),
        'manifest_sha256': digest(args.manifest), 'configuration': config,
        'configuration_sha256': hashlib.sha256(yaml.safe_dump(config).encode()).hexdigest(),
        'binary_road': road_summary(mask_counts) if mask_counts else None,
        'detection': coco_detection_metrics(images, annotations, predictions, category_names) if annotations else None,
        'label_schema': manifest.get('label_schema'), 'ground_truth_category_counts': dict(gt_counts),
        'road_observation_frames': dict(Counter(row['road_status'] for row in trace if 'road_status' in row)),
        'models': {'segformer_revision': getattr(lane.model.config, '_commit_hash', None) if lane else None,
                   'segformer_local_files_sha256': local_checkpoint,
                   'yolo_weights_sha256': digest(weights) if detector and weights.is_file() else None},
        'environment': {'device': device, 'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                        'python': platform.python_version(),
                        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'ultralytics', 'numpy')}},
        'source_files_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
            [Path(__file__).resolve(), *sorted((ROOT / 'src').rglob('*.py'))]},
        'samples': trace,
        'limitations': ['Binary road/drivable IoU is not multiclass mIoU or lane-marking accuracy',
            'Runtime postprocessing and deployed confidence thresholds are retained; AP is not a low-confidence raw-model validation',
            'No training occurs here; val/test must not be used to tune the reported configuration',
            'Small held-out subsets do not establish driving safety or full Indian-road generalization',
            'No detection AP is available for a dataset with only semantic labels'],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ('images', 'binary_road', 'detection')}, indent=2))
    print(f'Report: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/idd_eval.yaml'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default=str(ROOT / 'runs/accuracy/result.json'))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    evaluate(args)


if __name__ == '__main__':
    main()
