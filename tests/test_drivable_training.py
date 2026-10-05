"""Training/split contracts on generated fixtures, not claimed model accuracy."""
import copy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.prepare_idd_lite_training import check_separation, load_frozen, prepare
from scripts.train_drivable_idd_lite import (
    DrivableDataset,
    remap_labels,
    safe_optimizer_step,
    segmentation_loss,
    target_status,
)


def fixture_dataset(root):
    for index, (split, drive) in enumerate([('train', 'a'), ('train', 'b'), ('train', 'c'), ('val', 'v')]):
        images, masks = root / 'leftImg8bit' / split / drive, root / 'gtFine' / split / drive
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for frame in range(2):
            image = np.full((8, 12, 3), index * 50 + frame * 10, np.uint8)
            raw = np.full((8, 12), 3, np.uint8)
            raw[4:] = 0
            raw[0, 0] = 255
            assert cv2.imwrite(str(images / f'{frame}_image.jpg'), image)
            assert cv2.imwrite(str(masks / f'{frame}_label.png'), raw)
            assert cv2.imwrite(str(masks / f'{frame}_inst_label.png'), np.ones_like(raw))


def test_training_manifest_is_seeded_drive_disjoint_and_freezes_inputs(tmp_path):
    fixture_dataset(tmp_path)
    first = prepare(tmp_path, tmp_path / 'one.json')
    second = prepare(tmp_path, tmp_path / 'two.json')
    assert first == second
    assert first == load_frozen(tmp_path / 'one.json', tmp_path)
    assert first['counts'] == {'fit': {'images': 4, 'drives': 2},
                               'development': {'images': 2, 'drives': 1},
                               'official_val': {'images': 2, 'drives': 1}}
    assert all('train/' in row['image'] for key in ('fit', 'development') for row in first['splits'][key])
    with pytest.raises(ValueError, match='overwrite'):
        prepare(tmp_path, tmp_path / 'one.json')


@pytest.mark.parametrize('key', ['drive', 'image_sha256', 'decoded_image_sha256'])
def test_cross_split_leakage_fails(tmp_path, key):
    fixture_dataset(tmp_path)
    manifest = prepare(tmp_path, tmp_path / 'manifest.json')
    splits = copy.deepcopy(manifest['splits'])
    splits['development'][0][key] = splits['fit'][0][key]
    with pytest.raises(ValueError, match='leakage'):
        check_separation(splits)


def test_modified_frozen_mask_fails(tmp_path):
    fixture_dataset(tmp_path)
    manifest = prepare(tmp_path, tmp_path / 'manifest.json')
    mask = tmp_path / manifest['splits']['fit'][0]['mask']
    assert cv2.imwrite(str(mask), np.zeros((8, 12), np.uint8))
    with pytest.raises(ValueError, match='Changed frozen mask'):
        load_frozen(tmp_path / 'manifest.json', tmp_path)


def test_missing_training_mask_is_not_silently_background(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / 'gtFine/train/a/0_label.png').unlink()
    with pytest.raises(ValueError, match='Unpaired'):
        prepare(tmp_path, tmp_path / 'manifest.json')


def test_wrong_official_split_paths_fail(tmp_path):
    fixture_dataset(tmp_path)
    manifest = prepare(tmp_path, tmp_path / 'manifest.json')
    manifest['splits']['fit'][0]['image'] = 'leftImg8bit/val/v/0_image.jpg'
    (tmp_path / 'changed.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Wrong official split'):
        load_frozen(tmp_path / 'changed.json', tmp_path)


def test_label_zero_is_drivable_and_ignore_is_not_background():
    raw = np.array([[0, 1, 6, 255]], np.uint8)
    assert remap_labels(raw).tolist() == [[1, 0, 0, 255]]
    with pytest.raises(ValueError, match='level1Id'):
        remap_labels(np.array([[7]], np.uint8))


def test_ignore_pixels_have_zero_loss_gradient_and_valid_pixels_train():
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    labels = torch.tensor([[[0, 1], [255, 1]]])
    loss = segmentation_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(logits.grad[0, :, 1, 0] == 0)
    assert torch.any(logits.grad[0, :, 0, 1] != 0)
    with pytest.raises(ValueError, match='no valid'):
        segmentation_loss(logits, torch.full_like(labels, 255))


def test_dataset_retains_native_label_resolution_and_ignore(tmp_path):
    fixture_dataset(tmp_path)
    manifest = prepare(tmp_path, tmp_path / 'manifest.json')
    processor = SimpleNamespace(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    dataset = DrivableDataset(tmp_path, manifest['splits']['fit'], processor, 64)
    pixels, labels = dataset[0]
    assert pixels.shape == (3, 64, 64)
    assert labels.shape == (8, 12)
    assert labels[0, 0] == 255
    assert labels[7, 0] == 1


def test_all_user_targets_must_be_met_independently():
    assert not all(target_status({'IoU': 0.9319, 'precision': 0.98, 'recall': 0.95}).values())
    assert all(target_status({'IoU': 0.95, 'precision': 0.98, 'recall': 0.969}).values())
    assert not any(target_status({'IoU': None, 'precision': None, 'recall': None}).values())


def test_half_precision_logits_are_resized_in_float32(monkeypatch):
    from scripts import train_drivable_idd_lite as trainer
    original = trainer.F.interpolate
    seen = []

    def checked_interpolate(values, *args, **kwargs):
        seen.append(values.dtype)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(trainer.F, 'interpolate', checked_interpolate)
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float16, requires_grad=True)
    loss = segmentation_loss(logits, torch.tensor([[[0, 1, 1], [255, 0, 1], [0, 1, 1]]]))
    loss.backward()
    assert seen == [torch.float32]
    assert loss.dtype == torch.float32
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('bad_gradient', [float('inf'), float('nan')])
def test_real_scaler_skips_overflow_preserves_optimizer_and_scheduler_then_recovers(bad_gradient):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    # CPU GradScaler tests the real public overflow/recovery API without a GPU.
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0, growth_interval=2)
    scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(bad_gradient)
    before = parameter.detach().clone()
    scheduler_before = scheduler.state_dict().copy()
    assert not safe_optimizer_step([parameter], optimizer, scaler, scheduler)
    assert torch.equal(parameter, before)
    assert not optimizer.state
    assert scheduler.state_dict() == scheduler_before
    assert scaler.get_scale() == 4.0
    assert parameter.grad is None
    scaler.scale(parameter.sum()).backward()
    assert safe_optimizer_step([parameter], optimizer, scaler, scheduler)
    assert not torch.equal(parameter, before)
    assert torch.isfinite(parameter).all()
    assert scheduler.last_epoch == scheduler_before['last_epoch'] + 1


def test_finite_step_clips_unscaled_gradients_before_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0)
    scaler.scale(100 * parameter.sum()).backward()
    assert safe_optimizer_step([parameter], optimizer, scaler, scheduler)
    assert parameter.item() == pytest.approx(0.9)
    assert scheduler.last_epoch == 1


def test_fp32_nonfinite_gradients_fail_without_updating_parameters():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    parameter.grad = torch.tensor([float('inf')])
    with pytest.raises(RuntimeError, match='without AMP'):
        safe_optimizer_step([parameter], optimizer, scaler, scheduler)
    assert parameter.item() == 1.0
    assert scheduler.last_epoch == 0


def test_missing_gradients_fail_instead_of_counting_an_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    with pytest.raises(ValueError, match='No gradients'):
        safe_optimizer_step([parameter], optimizer, torch.amp.GradScaler('cpu'), scheduler)
