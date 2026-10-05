"""Freeze drive-disjoint fit/development data, leaving official val out of training."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root, split):
    mask_root = root / 'gtFine' / split
    image_root = root / 'leftImg8bit' / split
    masks = sorted(p for p in mask_root.glob('*/*_label.png') if not p.name.endswith('_inst_label.png'))
    if not masks:
        raise ValueError(f'No official {split} image/mask pairs')
    samples = []
    for mask in masks:
        image = image_root / mask.parent.name / (mask.name.removesuffix('_label.png') + '_image.jpg')
        frame, labels = cv2.imread(str(image)), cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if frame is None or labels is None or labels.ndim != 2 or labels.shape != frame.shape[:2]:
            raise ValueError(f'Missing or mismatched image/mask: {image}')
        if not set(np.unique(labels)) <= set(range(7)) | {255} or np.all(labels == 255):
            raise ValueError(f'Invalid level1Id labels: {mask}')
        samples.append({'id': f'{split}/{mask.parent.name}/{mask.stem}', 'drive': mask.parent.name,
                        'image': image.relative_to(root).as_posix(), 'mask': mask.relative_to(root).as_posix(),
                        'image_sha256': digest(image), 'mask_sha256': digest(mask),
                        'decoded_image_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                        'resolution': [frame.shape[1], frame.shape[0]]})
    if {root / row['image'] for row in samples} != set(image_root.glob('*/*_image.jpg')):
        raise ValueError(f'Unpaired {split} images')
    return samples


def check_separation(splits):
    seen_drives, seen_images, seen_decoded = set(), set(), set()
    for name in ('fit', 'development', 'official_val'):
        rows = splits[name]
        if not rows:
            raise ValueError(f'Empty {name} split')
        drives = {row['drive'] for row in rows}
        images = {row['image_sha256'] for row in rows}
        decoded = {row['decoded_image_sha256'] for row in rows}
        if seen_drives & drives or seen_images & images or seen_decoded & decoded:
            raise ValueError('Drive or exact/decoded duplicate image leakage across splits')
        seen_drives |= drives
        seen_images |= images
        seen_decoded |= decoded


def prepare(root, output, seed=42, development_fraction=0.15):
    if not 0 < development_fraction < 1:
        raise ValueError('development_fraction must be between 0 and 1')
    root, output = Path(root).resolve(), Path(output)
    train, official_val = collect(root, 'train'), collect(root, 'val')
    drives = sorted({row['drive'] for row in train})
    if len(drives) < 2:
        raise ValueError('Need at least two training drive folders')
    shuffled = random.Random(seed).sample(drives, len(drives))
    count = min(len(drives) - 1, max(1, round(len(drives) * development_fraction)))
    dev_drives = set(shuffled[:count])
    splits = {'fit': [row for row in train if row['drive'] not in dev_drives],
              'development': [row for row in train if row['drive'] in dev_drives],
              'official_val': official_val}
    check_separation(splits)
    manifest = {'schema_version': 1, 'dataset': 'official IDD Lite', 'seed': seed,
                'development_fraction_of_drives': development_fraction,
                'label_schema': {'raw_valid_ids': list(range(7)), 'raw_drivable_id': 0,
                                 'training_non_drivable_id': 0, 'training_drivable_id': 1, 'ignore_id': 255},
                'policy': 'Development drive folders sampled from official train before training; official val never used for gradient updates or checkpoint selection.',
                'counts': {name: {'images': len(rows), 'drives': len({row['drive'] for row in rows})}
                           for name, rows in splits.items()},
                'limitations': ['Folder grouping and exact/decoded duplicate checks do not establish absence of near-duplicates or independent geographic routes.',
                                'Official val already has a published baseline and is not a never-seen test set.'],
                'splits': splits}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen split manifest')
    output.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_frozen(path, root):
    manifest = json.loads(Path(path).read_text())
    root = Path(root).resolve()
    schema = {'raw_valid_ids': list(range(7)), 'raw_drivable_id': 0,
              'training_non_drivable_id': 0, 'training_drivable_id': 1, 'ignore_id': 255}
    if manifest.get('schema_version') != 1 or manifest.get('label_schema') != schema:
        raise ValueError('Unsupported training manifest or label schema')
    splits = manifest['splits']
    check_separation(splits)
    ids = [row['id'] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs')
    for name, rows in splits.items():
        if name not in ('fit', 'development', 'official_val'):
            raise ValueError('Unexpected split')
        expected = 'val' if name == 'official_val' else 'train'
        for row in rows:
            for key, prefix in (('image', 'leftImg8bit'), ('mask', 'gtFine')):
                relative = Path(row[key])
                if relative.is_absolute() or relative.parts[:3] != (prefix, expected, row['drive']):
                    raise ValueError('Wrong official split or drive path')
                actual = (root / relative).resolve()
                actual.relative_to(root)
                if digest(actual) != row[key + '_sha256']:
                    raise ValueError(f'Changed frozen {key}: {actual}')
            frame = cv2.imread(str(root / row['image']))
            if frame is None or hashlib.sha256(frame.tobytes()).hexdigest() != row['decoded_image_sha256']:
                raise ValueError('Changed decoded image')
        if manifest['counts'][name] != {'images': len(rows), 'drives': len({row['drive'] for row in rows})}:
            raise ValueError('Wrong manifest split counts')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--development-fraction', type=float, default=0.15)
    args = parser.parse_args()
    result = prepare(args.root, args.output, args.seed, args.development_fraction)
    print(json.dumps(result['counts'], indent=2))
    print('Frozen manifest SHA-256:', digest(args.output))


if __name__ == '__main__':
    main()
