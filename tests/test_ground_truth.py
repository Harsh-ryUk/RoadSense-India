"""Synthetic fixtures validate metric arithmetic, not project accuracy."""
import json

import numpy as np
import pytest

from scripts.evaluate_accuracy import load_manifest, local_segformer_fingerprint
from src.evaluation.ground_truth import binary_road_counts, coco_detection_metrics, road_summary


def test_ignored_pixels_do_not_count_as_false_positives():
    labels = np.array([[0, 1], [0, 255]], np.uint8)
    predicted = np.array([[255, 255], [0, 255]], np.uint8)
    counts = binary_road_counts(predicted, labels, [0], [0, 1], [255])
    assert counts == dict(tp=1, fp=1, fn=1, tn=0, valid_pixels=3, ignored_pixels=1)
    assert road_summary([counts])['IoU'] == pytest.approx(1 / 3)


def test_empty_road_prediction_keeps_all_missed_gt_pixels():
    labels = np.array([[0, 0], [1, 1]], np.uint8)
    counts = binary_road_counts(np.zeros_like(labels), labels, [0], [0, 1], [255])
    assert counts['fn'] == 2
    assert road_summary([counts])['IoU'] == 0
    assert road_summary([counts])['precision'] is None


def test_road_summary_uses_global_counts_not_image_average():
    a = binary_road_counts(np.ones((1, 1)), np.zeros((1, 1)), [0], [0, 1], [255])
    b = binary_road_counts(np.zeros((3, 3)), np.zeros((3, 3)), [0], [0, 1], [255])
    assert road_summary([a, b])['IoU'] == pytest.approx(0.1)


def test_mismatched_or_unknown_mask_labels_fail():
    with pytest.raises(ValueError, match='shape'):
        binary_road_counts(np.zeros((2, 2)), np.zeros((1, 2)), [0], [0, 1], [255])
    with pytest.raises(ValueError, match='Unexpected'):
        binary_road_counts(np.zeros((1, 2)), np.array([[7, 0]]), [0], [0, 1], [255])
    with pytest.raises(ValueError, match='No valid'):
        road_summary([binary_road_counts(np.zeros((1, 1)), np.full((1, 1), 255), [0], [0, 1], [255])])


def test_manifest_rejects_training_split_and_missing_ground_truth(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema_version': 1, 'split': 'train', 'samples': []}))
    with pytest.raises(ValueError, match='held-out'):
        load_manifest(path)
    path.write_text(json.dumps({'schema_version': 1, 'split': 'val', 'samples': [{'id': 'a', 'image': 'a.png'}]}))
    with pytest.raises(ValueError, match='mask'):
        load_manifest(path)


def test_coco_ap_includes_gt_class_with_no_predictions():
    pytest.importorskip('pycocotools')
    images = [{'id': 1, 'width': 50, 'height': 50}]
    gt = [{'id': index, 'image_id': 1, 'category_id': index,
           'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0} for index in (1, 2)]
    predictions = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': 0.9}]
    metrics = coco_detection_metrics(images, gt, predictions, ['car', 'autorickshaw'])
    assert metrics['mAP50_95'] == pytest.approx(0.5)
    assert metrics['per_class_AP50_95']['autorickshaw'] == 0
    assert coco_detection_metrics(images, gt, [], ['car', 'autorickshaw'])['mAP50_95'] == 0


def test_local_fine_tuned_checkpoint_fingerprint_tracks_weight_changes(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'generated fixture, not actual weights')
    first = local_segformer_fingerprint(str(tmp_path))
    weights.write_bytes(b'changed fixture')
    second = local_segformer_fingerprint(str(tmp_path))
    assert first['config.json'] == second['config.json']
    assert first['model.safetensors'] != second['model.safetensors']
    assert local_segformer_fingerprint('nvidia/segformer-b0-finetuned-ade-512-512') is None
