"""Generated fixtures only: these tests are not measured Indian-road accuracy."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.evaluate_accuracy import check_evaluation_contract, load_manifest
from scripts.prepare_independent_road_test import (
    TAGS,
    choose_frames,
    digest,
    freeze,
    read_sessions,
    sample,
    validate_mask,
    validate_reviews,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / 'best'
    checkpoint.mkdir()
    write_json(checkpoint / 'config.json', {'id2label': {'0': 'non_drivable', '1': 'drivable'}})
    write_json(checkpoint / 'preprocessor_config.json', {'do_reduce_labels': False})
    (checkpoint / 'model.safetensors').write_bytes(b'FAKE CHECKPOINT FOR CONTRACT TESTS ONLY')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'segmentation': {'model_name': str(checkpoint),
        'input_size': [512, 512], 'road_class_ids': [1], 'frame_roi': [0, 0, 1, 1],
        'roi_top_fraction': 0, 'min_road_coverage': 0.02}}))
    training = tmp_path / 'splits.json'
    write_json(training, {'schema_version': 1, 'splits': {
        name: [{'image_sha256': str(i) * 64, 'decoded_image_sha256': str(i) * 64}]
        for i, name in enumerate(('fit', 'development', 'official_val'), 1)}})
    rows = []
    for i in range(2):
        video = tmp_path / f'drive_{i}.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5.0, (64, 48))
        assert writer.isOpened(), 'Generated fixture video codec unavailable'
        rng = np.random.default_rng(i + 101)
        for _ in range(40):
            writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        writer.release()
        rows.append({'id': f'drive_{i}', 'session_group': f'session_{i}', 'video': video.name,
                     'permission_confirmed': True, 'unused_for_development': True})
    sessions = tmp_path / 'sessions.json'
    write_json(sessions, {'schema_version': 1, 'sessions': rows})
    return {'sessions_path': sessions, 'training_manifest': training, 'checkpoint': checkpoint,
            'config_path': config, 'output': tmp_path / 'test_package', 'count': 3}


def annotate(package):
    plan = json.loads((package / 'frame_plan.json').read_text())
    reviews = json.loads((package / 'annotation_reviews.json').read_text())
    for row, review in zip(plan['samples'], reviews['samples']):
        width, height = row['resolution']
        mask = np.zeros((height, width), np.uint8)
        mask[height // 2:, :] = 1
        mask[0, 0] = 255
        Image.fromarray(mask).save(package / row['mask'])
        review.update(human_reviewed=True, annotator_alias='fixture_annotator',
                      reviewer_alias='fixture_reviewer', mask_sha256=digest(package / row['mask']),
                      tags={tag: 'absent' for tag in TAGS})
    write_json(package / 'annotation_reviews.json', reviews)
    return plan, reviews


def freeze_package(package):
    return freeze(package / 'frame_plan.json', package / 'annotation_reviews.json', package / 'manifest.json')


def test_sample_annotate_freeze_and_evaluator_contract(inputs):
    plan = sample(**inputs)
    package = inputs['output']
    assert len(plan['samples']) == 6
    assert plan['stage'] == 'awaiting_human_annotations'
    assert not list((package / 'masks').rglob('*.png'))  # No fabricated all-zero GT.
    assert [row['source_frame_index'] for row in plan['samples'][:3]] == [9, 19, 29]
    assert all(row['resolution'] == [64, 48] for row in plan['samples'])
    assert len({row['decoded_image_sha256'] for row in plan['samples']}) == 6
    annotate(package)
    result = freeze_package(package)
    assert result['accuracy'] is None
    assert result['split'] == 'test' and result['stage'] == 'frozen_human_reviewed_test'
    assert result['label_schema'] == {'road_ids': [1], 'valid_ids': [0, 1], 'ignore_ids': [255]}
    assert len(load_manifest(package / 'manifest.json')['samples']) == 6
    check_evaluation_contract(result, package / 'evaluation_config.yaml', str(inputs['checkpoint']))
    with pytest.raises(ValueError, match='overwrite'):
        freeze_package(package)
    with pytest.raises(FileExistsError):
        sample(**inputs)


@pytest.mark.parametrize('fps,gap,count,total', [
    (float('nan'), 2, 3, 100), (0, 2, 3, 100), (5, -1, 3, 100),
    (5, 2, 1, 100), (5, 2, 3, 10), (5, 2, 1000, 100),
])
def test_sampling_rejects_invalid_or_dense_frames(fps, gap, count, total):
    with pytest.raises(ValueError):
        choose_frames(total, fps, count, gap)


@pytest.mark.parametrize('mutation', ['used', 'rights', 'same_group', 'same_video', 'unsafe_id', 'missing_video'])
def test_ineligible_sessions_rejected(inputs, mutation):
    path = inputs['sessions_path']
    data = json.loads(path.read_text())
    if mutation == 'used':
        data['sessions'][0]['unused_for_development'] = False
    elif mutation == 'rights':
        data['sessions'][0]['permission_confirmed'] = False
    elif mutation == 'same_group':
        data['sessions'][1]['session_group'] = data['sessions'][0]['session_group']
    elif mutation == 'same_video':
        data['sessions'][1]['video'] = data['sessions'][0]['video']
    elif mutation == 'unsafe_id':
        data['sessions'][0]['id'] = '../escape'
    else:
        data['sessions'][0]['video'] = 'missing.avi'
    write_json(path, data)
    with pytest.raises(ValueError):
        read_sessions(path)


def test_duplicate_video_bytes_rejected(inputs):
    sessions = read_sessions(inputs['sessions_path'])
    Path(sessions[1]['video_path']).write_bytes(Path(sessions[0]['video_path']).read_bytes())
    with pytest.raises(ValueError, match='Duplicate source'):
        sample(**inputs)
    assert not inputs['output'].exists()


def test_training_decoded_duplicate_rejected(inputs):
    source = read_sessions(inputs['sessions_path'])[0]['video_path']
    cap = cv2.VideoCapture(source)
    for _ in range(10):
        ok, frame = cap.read()
        assert ok
    cap.release()
    training = json.loads(inputs['training_manifest'].read_text())
    training['splits']['fit'][0]['decoded_image_sha256'] = hashlib.sha256(frame.tobytes()).hexdigest()
    write_json(inputs['training_manifest'], training)
    with pytest.raises(ValueError, match='duplicate'):
        sample(**inputs)
    assert not (inputs['output'] / 'frame_plan.json').exists()


def test_missing_annotations_never_become_a_test(inputs):
    plan = sample(**inputs)
    reviews = json.loads((inputs['output'] / 'annotation_reviews.json').read_text())
    with pytest.raises(ValueError, match='Unreviewed'):
        validate_reviews(reviews, plan['samples'])
    with pytest.raises(ValueError, match='Unreviewed'):
        freeze_package(inputs['output'])


@pytest.mark.parametrize('mutation', ['dropped', 'duplicate', 'tag', 'reviewer', 'review_hash'])
def test_annotation_reviews_are_complete_and_bound(inputs, mutation):
    sample(**inputs)
    package = inputs['output']
    plan, reviews = annotate(package)
    reviews = copy.deepcopy(reviews)
    if mutation == 'dropped':
        reviews['samples'].pop()
    elif mutation == 'duplicate':
        reviews['samples'][1]['id'] = reviews['samples'][0]['id']
    elif mutation == 'tag':
        reviews['samples'][0]['tags']['rain'] = 'maybe'
    elif mutation == 'reviewer':
        reviews['samples'][0]['reviewer_alias'] = ''
    else:
        reviews['samples'][0]['mask_sha256'] = ''
    with pytest.raises(ValueError):
        validate_reviews(reviews, plan['samples'])


@pytest.mark.parametrize('mutation', ['rgb', 'wrong_size', 'wrong_id', 'all_ignored'])
def test_invalid_masks_rejected(tmp_path, mutation):
    shape = (4, 6, 3) if mutation == 'rgb' else (4, 6)
    pixels = np.zeros(shape, np.uint8)
    if mutation == 'wrong_id':
        pixels[0, 0] = 7
    elif mutation == 'all_ignored':
        pixels[:] = 255
    mask = tmp_path / 'mask.png'
    Image.fromarray(pixels).save(mask)
    with pytest.raises(ValueError):
        validate_mask(mask, [6, 5] if mutation == 'wrong_size' else [6, 4])


def test_negative_mask_is_retained(tmp_path):
    mask = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((4, 6), np.uint8)).save(mask)
    assert validate_mask(mask, [6, 4]) == {'valid_pixels': 24, 'ignored_pixels': 0}


@pytest.mark.parametrize('mutation', ['model', 'config', 'mask', 'image', 'training', 'unsafe_mask'])
def test_freeze_rejects_changed_inputs(inputs, mutation):
    sample(**inputs)
    package = inputs['output']
    plan, _ = annotate(package)
    if mutation == 'model':
        (inputs['checkpoint'] / 'model.safetensors').write_bytes(b'changed')
    elif mutation == 'config':
        with (package / 'evaluation_config.yaml').open('a') as handle:
            handle.write('\n# changed after sampling\n')
    elif mutation == 'mask':
        Image.fromarray(np.ones((48, 64), np.uint8)).save(package / plan['samples'][0]['mask'])
    elif mutation == 'image':
        Image.fromarray(np.ones((48, 64, 3), np.uint8)).save(package / plan['samples'][0]['image'])
    elif mutation == 'training':
        inputs['training_manifest'].write_text('{}')
    else:
        plan['samples'][0]['mask'] = '../outside.png'
        write_json(package / 'frame_plan.json', plan)
    with pytest.raises(ValueError):
        freeze_package(package)
    assert not (package / 'manifest.json').exists()


@pytest.mark.parametrize('mutation', ['model', 'config', 'stage', 'hub'])
def test_evaluator_rejects_model_config_or_stage_drift(inputs, mutation):
    sample(**inputs)
    package = inputs['output']
    annotate(package)
    manifest = freeze_package(package)
    model = str(inputs['checkpoint'])
    if mutation == 'model':
        (inputs['checkpoint'] / 'model.safetensors').write_bytes(b'changed')
    elif mutation == 'config':
        with (package / 'evaluation_config.yaml').open('a') as handle:
            handle.write('\n# changed\n')
    elif mutation == 'stage':
        manifest['stage'] = 'awaiting_human_annotations'
    else:
        model = 'nvidia/segformer-b0-finetuned-ade-512-512'
    with pytest.raises(ValueError):
        check_evaluation_contract(manifest, package / 'evaluation_config.yaml', model)


def test_original_idd_contract_remains_supported(tmp_path):
    check_evaluation_contract({'dataset': 'official IDD Lite'}, tmp_path / 'not-needed.yaml', 'hub-model')


def test_sample_and_freeze_cli(inputs):
    script = Path(__file__).resolve().parents[1] / 'scripts/prepare_independent_road_test.py'
    command = [sys.executable, str(script), 'sample', '--sessions', str(inputs['sessions_path']),
               '--training-manifest', str(inputs['training_manifest']), '--checkpoint', str(inputs['checkpoint']),
               '--config', str(inputs['config_path']), '--output', str(inputs['output']), '--frames-per-session', '3']
    prepared = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(prepared.stdout) == {'stage': 'awaiting_human_annotations', 'images': 6, 'accuracy': None}
    package = inputs['output']
    annotate(package)
    frozen = subprocess.run([sys.executable, str(script), 'freeze', '--plan', str(package / 'frame_plan.json'),
                             '--reviews', str(package / 'annotation_reviews.json'),
                             '--output', str(package / 'manifest.json')], check=True, capture_output=True, text=True)
    assert json.loads(frozen.stdout) == {'stage': 'frozen_human_reviewed_test', 'images': 6, 'accuracy': None}


def test_undecodable_video_does_not_produce_a_plan(inputs):
    source = read_sessions(inputs['sessions_path'])[0]['video_path']
    Path(source).write_bytes(b'not a video')
    with pytest.raises(ValueError, match='Cannot open video'):
        sample(**inputs)
    assert not (inputs['output'] / 'frame_plan.json').exists()
