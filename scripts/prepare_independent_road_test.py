"""Sample new road sessions before inference, then freeze human-reviewed masks.

This prepares an independent test; it neither trains a model nor invents labels.
See docs/INDEPENDENT_ROAD_TEST.md for the two-stage workflow.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

TAGS = ('shadows', 'heavy_traffic', 'poor_boundaries', 'rain', 'night')
TAG_VALUES = ('present', 'absent', 'uncertain')
SCHEMA = {'road_ids': [1], 'valid_ids': [0, 1], 'ignore_ids': [255]}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def checkpoint_fingerprint(directory):
    directory = Path(directory).resolve()
    required = ('config.json', 'preprocessor_config.json', 'model.safetensors')
    if any(not (directory / name).is_file() for name in required):
        raise ValueError('Need the exported best checkpoint, config and preprocessor files')
    config = json.loads((directory / 'config.json').read_text())
    if set(config.get('id2label', {})) != {'0', '1'}:
        raise ValueError('Expected the trained two-class drivable checkpoint')
    paths = {directory / name for name in required}
    paths |= set(directory.glob('*.safetensors')) | set(directory.glob('pytorch_model*.bin'))
    paths |= set(directory.glob('*.index.json'))
    return {path.name: digest(path) for path in sorted(paths)}


def read_sessions(path):
    path = Path(path).resolve()
    source = json.loads(path.read_text())
    rows = source.get('sessions', [])
    if source.get('schema_version') != 1 or len(rows) < 2:
        raise ValueError('Provide at least two genuinely different road sessions')
    ids, groups, videos = set(), set(), set()
    result = []
    for row in rows:
        identity, group = row.get('id', ''), row.get('session_group', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity):
            raise ValueError('Session IDs must be short, path-safe names')
        if not isinstance(group, str) or not group.strip() or identity in ids or group in groups:
            raise ValueError('Duplicate or missing session group; cuts of one drive are not independent sessions')
        if row.get('permission_confirmed') is not True or row.get('unused_for_development') is not True:
            raise ValueError('Confirm footage permission and that the session was not used for development')
        video = (path.parent / row['video']).resolve()
        if not video.is_file() or video in videos:
            raise ValueError('Video missing or reused in more than one session')
        ids.add(identity)
        groups.add(group)
        videos.add(video)
        result.append({**row, 'video_path': str(video)})
    return result


def choose_frames(frame_count, fps, count, min_gap_seconds=2.0):
    if type(frame_count) is not int or frame_count < 1 or type(count) is not int or count < 2:
        raise ValueError('Need valid frame counts and at least two samples per session')
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(min_gap_seconds) or min_gap_seconds <= 0:
        raise ValueError('FPS and minimum spacing must be finite and positive')
    indices = [((i + 1) * (frame_count - 1)) // (count + 1) for i in range(count)]
    if len(set(indices)) != count or min(b - a for a, b in pairwise(indices)) / fps < min_gap_seconds:
        raise ValueError('Clip is too short for this many spaced samples; lower the count, not the spacing')
    return indices


def training_exclusions(path):
    reference = json.loads(Path(path).read_text())
    splits = reference.get('splits', {})
    if reference.get('schema_version') != 1 or set(splits) != {'fit', 'development', 'official_val'}:
        raise ValueError('Use the original frozen IDD fit/development/official_val manifest')
    encoded, decoded = set(), set()
    for rows in splits.values():
        if not rows:
            raise ValueError('Training reference has an empty split')
        for row in rows:
            for key, target in (('image_sha256', encoded), ('decoded_image_sha256', decoded)):
                value = row.get(key, '')
                if not re.fullmatch(r'[0-9a-f]{64}', value):
                    raise ValueError('Training reference is missing a valid image hash')
                target.add(value)
    return encoded, decoded


def frozen_configuration(config_path, checkpoint):
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    seg = config.get('segmentation', {}) if isinstance(config, dict) else {}
    if Path(seg.get('model_name', '')).resolve() != Path(checkpoint).resolve():
        raise ValueError('Evaluation config must refer to this exact local checkpoint')
    expected = {'road_class_ids': [1], 'input_size': [512, 512], 'frame_roi': [0, 0, 1, 1],
                'roi_top_fraction': 0, 'min_road_coverage': 0.02}
    if any(seg.get(key) != value for key, value in expected.items()):
        raise ValueError('Retain the measured 512-pixel, full-frame class-1 postprocessing configuration')
    return {'configuration_file_sha256': digest(config_path),
            'segformer_local_files_sha256': checkpoint_fingerprint(checkpoint)}


def sample(sessions_path, training_manifest, checkpoint, config_path, output, count=20, min_gap_seconds=2.0):
    import cv2
    from PIL import Image

    sessions_path, training_manifest = Path(sessions_path).resolve(), Path(training_manifest).resolve()
    checkpoint, config_path = Path(checkpoint).resolve(), Path(config_path).resolve()
    output = Path(output).resolve()
    sessions = read_sessions(sessions_path)
    excluded_encoded, excluded_decoded = training_exclusions(training_manifest)
    contract = frozen_configuration(config_path, checkpoint)
    source_hashes = [digest(row['video_path']) for row in sessions]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError('Duplicate source video content')
    output.mkdir(parents=True, exist_ok=False)  # Partial attempts are retained, never silently replaced.
    rows, seen_decoded, source_rows = [], set(), []
    reference_hash, sessions_hash = digest(training_manifest), digest(sessions_path)
    for session, video_hash in zip(sessions, source_hashes):
        cap = cv2.VideoCapture(session['video_path'])
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open video: ' + session['id'])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            selected = choose_frames(total, fps, count, min_gap_seconds)
            selected_set, captured = set(selected), set()
            image_dir, mask_dir = output / 'images' / session['id'], output / 'masks' / session['id']
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            # Decode sequentially; do not silently substitute an easier frame after a seek/decode failure.
            for index in range(selected[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Video decode failed at frame {index}: {session["id"]}')
                if index not in selected_set:
                    continue
                identity = f'{session["id"]}/frame_{index:08d}'
                decoded_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                if decoded_hash in excluded_decoded or decoded_hash in seen_decoded:
                    raise ValueError('Exact decoded duplicate in training or test samples: ' + identity)
                image = image_dir / f'frame_{index:08d}.png'
                Image.fromarray(frame[:, :, ::-1]).save(image)
                if digest(image) in excluded_encoded:
                    raise ValueError('Encoded duplicate in training: ' + identity)
                seen_decoded.add(decoded_hash)
                captured.add(index)
                rows.append({'id': identity, 'drive': session['id'], 'session_group': session['session_group'],
                             'image': image.relative_to(output).as_posix(),
                             'mask': f'masks/{session["id"]}/frame_{index:08d}.png',
                             'image_sha256': digest(image), 'decoded_image_sha256': decoded_hash,
                             'resolution': [frame.shape[1], frame.shape[0]],
                             'source_frame_index': index, 'nominal_time_seconds': index / fps})
            if captured != selected_set or digest(session['video_path']) != video_hash:
                raise ValueError('Incomplete sampling or source video changed')
            source_rows.append({'id': session['id'], 'session_group': session['session_group'],
                                'video_path': session['video_path'], 'video_sha256': video_hash,
                                'reported_frame_count': total, 'reported_fps': fps,
                                'selected_frame_indices': selected,
                                'permission_confirmed': True, 'unused_for_development': True})
        finally:
            cap.release()
    if contract != frozen_configuration(config_path, checkpoint) or reference_hash != digest(training_manifest) or sessions_hash != digest(sessions_path):
        raise ValueError('Model, configuration or source declarations changed during sampling')
    plan = {'schema_version': 1, 'stage': 'awaiting_human_annotations',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'independent Indian-road sessions', 'split': 'test', 'label_schema': SCHEMA,
            'selection': {'policy': 'uniformly spaced frame indices frozen before model predictions',
                          'frames_per_session': count, 'min_gap_seconds': min_gap_seconds,
                          'selected_images': len(rows)},
            'training_manifest_path': str(training_manifest), 'training_manifest_sha256': reference_hash,
            'sessions_declaration_sha256': sessions_hash, 'sources': source_rows,
            'checkpoint_path': str(checkpoint), 'evaluation_contract': contract, 'samples': rows,
            'limitations': ['Session independence and usage permission are human declarations, not programmatic proof.',
                            'Exact hash checks do not rule out near-duplicates or shared geographic routes.',
                            'Times and spacing use frame indices and reported FPS, not verified variable-rate presentation timestamps.',
                            'Frames within a session are correlated; this small sample is not a driving-safety test.',
                            'No prediction, training, or accuracy measurement has occurred in this preparation step.']}
    write_new_json(output / 'frame_plan.json', plan)
    write_new_json(output / 'annotation_reviews.json', {'schema_version': 1, 'samples': [
        {'id': row['id'], 'image_sha256': row['image_sha256'], 'human_reviewed': False,
         'annotator_alias': '', 'reviewer_alias': '', 'mask_sha256': '',
         'tags': {tag: 'uncertain' for tag in TAGS}} for row in rows]})
    with (output / 'evaluation_config.yaml').open('xb') as handle:
        handle.write(config_path.read_bytes())
    return plan


def validate_reviews(reviews, rows):
    if reviews.get('schema_version') != 1 or not rows:
        raise ValueError('Invalid or empty annotation review')
    entries = reviews.get('samples', [])
    by_id = {entry['id']: entry for entry in entries}
    if len(by_id) != len(entries) or set(by_id) != {row['id'] for row in rows}:
        raise ValueError('Every frozen frame needs exactly one review; no dropping difficult frames')
    for row in rows:
        review = by_id[row['id']]
        if review.get('human_reviewed') is not True or review.get('image_sha256') != row['image_sha256']:
            raise ValueError('Unreviewed or changed image: ' + row['id'])
        if any(not isinstance(review.get(key), str) or not review[key].strip()
               for key in ('annotator_alias', 'reviewer_alias')):
            raise ValueError('Record annotator and reviewer aliases: ' + row['id'])
        if not re.fullmatch(r'[0-9a-f]{64}', review.get('mask_sha256', '')):
            raise ValueError('Review must bind the final mask SHA-256: ' + row['id'])
        if set(review.get('tags', {})) != set(TAGS) or any(value not in TAG_VALUES for value in review['tags'].values()):
            raise ValueError('Each scenario tag must be present, absent or uncertain')
    return by_id


def validate_mask(path, resolution):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'L' or list(image.size) != resolution:
            raise ValueError('Mask must be native-size, 8-bit grayscale PNG (not RGB, palette or resized)')
        pixels = np.array(image)
    if not set(np.unique(pixels).tolist()) <= {0, 1, 255} or not np.any(pixels != 255):
        raise ValueError('Mask IDs must be 0/1/255 with at least one valid pixel')
    return {'valid_pixels': int(np.count_nonzero(pixels != 255)),
            'ignored_pixels': int(np.count_nonzero(pixels == 255))}


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    if Path(relative).is_absolute() or not path.is_file():
        raise ValueError('Missing or unsafe package file: ' + str(relative))
    return path


def freeze(plan_path, reviews_path, output):
    import numpy as np
    from PIL import Image

    plan_path, reviews_path, output = Path(plan_path).resolve(), Path(reviews_path).resolve(), Path(output).resolve()
    root = plan_path.parent
    if output.parent != root:
        raise ValueError('Write the frozen manifest next to frame_plan.json to keep image paths valid')
    plan = json.loads(plan_path.read_text())
    if plan.get('schema_version') != 1 or plan.get('stage') != 'awaiting_human_annotations' or plan.get('label_schema') != SCHEMA:
        raise ValueError('Unsupported frame plan')
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen independent test')
    references = Path(plan['training_manifest_path'])
    if digest(references) != plan['training_manifest_sha256']:
        raise ValueError('Training exclusion manifest changed')
    encoded, decoded = training_exclusions(references)
    if frozen_configuration(root / 'evaluation_config.yaml', plan['checkpoint_path']) != plan['evaluation_contract']:
        raise ValueError('Frozen checkpoint or evaluation configuration changed')
    rows = plan['samples']
    if len({row['id'] for row in rows}) != len(rows) or len({row['session_group'] for row in rows}) < 2:
        raise ValueError('Need unique samples from at least two session groups')
    plan_hash, reviews_hash = digest(plan_path), digest(reviews_path)
    by_id = validate_reviews(json.loads(reviews_path.read_text()), rows)
    frozen, seen = [], set()
    for row in rows:
        image, mask = inside(root, row['image']), inside(root, row['mask'])
        if digest(image) != row['image_sha256'] or digest(image) in encoded:
            raise ValueError('Changed or reused test image: ' + row['id'])
        with Image.open(image) as source:
            if source.mode != 'RGB' or list(source.size) != row['resolution']:
                raise ValueError('Changed image dimensions or encoding')
            actual_decoded = hashlib.sha256(np.array(source)[:, :, ::-1].tobytes()).hexdigest()
        if actual_decoded != row['decoded_image_sha256'] or actual_decoded in decoded or actual_decoded in seen:
            raise ValueError('Changed or duplicate decoded image: ' + row['id'])
        seen.add(actual_decoded)
        accounting = validate_mask(mask, row['resolution'])
        mask_hash = digest(mask)
        if by_id[row['id']]['mask_sha256'] != mask_hash:
            raise ValueError('Mask changed after human review: ' + row['id'])
        frozen.append({**row, 'mask_sha256': mask_hash, 'annotation_accounting': accounting,
                       'tags': by_id[row['id']]['tags']})
    if digest(plan_path) != plan_hash or digest(reviews_path) != reviews_hash:
        raise ValueError('Plan or reviews changed during freezing')
    if digest(references) != plan['training_manifest_sha256'] or frozen_configuration(
            root / 'evaluation_config.yaml', plan['checkpoint_path']) != plan['evaluation_contract']:
        raise ValueError('Training reference, checkpoint or configuration changed during freezing')
    for row in frozen:
        if digest(root / row['image']) != row['image_sha256'] or digest(root / row['mask']) != row['mask_sha256']:
            raise ValueError('Image or mask changed during freezing')
    result = {**plan, 'stage': 'frozen_human_reviewed_test', 'samples': frozen,
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'frame_plan_sha256': plan_hash, 'annotation_reviews_sha256': reviews_hash,
              'annotation_policy': 'Human drawn/reviewed native-size masks; no model-generated masks accepted as ground truth.',
              'categories': [], 'accuracy': None}
    write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('sample')
    for name in ('sessions', 'training-manifest', 'checkpoint', 'config', 'output'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--frames-per-session', type=int, default=20)
    prep.add_argument('--min-gap-seconds', type=float, default=2.0)
    final = commands.add_parser('freeze')
    for name in ('plan', 'reviews', 'output'):
        final.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'sample':
        result = sample(args.sessions, args.training_manifest, args.checkpoint, args.config, args.output,
                        args.frames_per_session, args.min_gap_seconds)
    else:
        result = freeze(args.plan, args.reviews, args.output)
    print(json.dumps({'stage': result['stage'], 'images': len(result['samples']), 'accuracy': None}, indent=2))


if __name__ == '__main__':
    main()
