"""Independently audit saved binary segmentation counts; does not run inference."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


COUNT_KEYS = ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels')


def audit(report, manifest):
    rows, frozen = report['samples'], manifest['samples']
    if not rows or len(rows) != report['images'] or len(rows) != len(frozen):
        raise ValueError('Report and frozen sample counts do not agree')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    for row, expected in zip(rows, frozen):
        for key in ('id', 'image_sha256', 'mask_sha256'):
            if row[key] != expected[key]:
                raise ValueError(f'Frozen identity mismatch: {key}')
        if any(type(row[key]) is not int or row[key] < 0 for key in COUNT_KEYS):
            raise ValueError('Pixel counts must be nonnegative integers')
        if sum(row[key] for key in ('tp', 'fp', 'fn', 'tn')) != row['valid_pixels']:
            raise ValueError('Confusion counts do not account for every valid pixel')
        width, height = row['resolution']
        if width <= 0 or height <= 0 or row['valid_pixels'] + row['ignored_pixels'] != width * height:
            raise ValueError('Pixel accounting does not match image dimensions')
    totals = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    summary = report['binary_road']
    if totals['valid_pixels'] == 0 or any(summary[key] != totals[key] for key in COUNT_KEYS):
        raise ValueError('Aggregate counts differ from the per-image evidence')
    tp, fp, fn = (totals[key] for key in ('tp', 'fp', 'fn'))
    ratios = {'IoU': (tp, tp + fp + fn), 'precision': (tp, tp + fp), 'recall': (tp, tp + fn)}
    for key, (numerator, denominator) in ratios.items():
        expected = numerator / denominator if denominator else None
        actual = summary[key]
        if (expected is None and actual is not None) or (expected is not None and
                (actual is None or not math.isclose(actual, expected, rel_tol=1e-12))):
            raise ValueError(f'Incorrect aggregate {key}')
    statuses = dict(Counter(row['road_status'] for row in rows))
    if statuses != report['road_observation_frames']:
        raise ValueError('Road observation totals differ from the trace')
    return {'audit': 'passed', 'images': len(rows), 'drives': len({row['id'].split('/')[0] for row in rows}),
            'counts': totals, 'metrics': {key: summary[key] for key in ratios}, 'road_status': statuses,
            'scope': 'identity and arithmetic only; not a second model run or human annotation audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(Path(args.report).read_text()),
                           json.loads(Path(args.manifest).read_text())), indent=2))


if __name__ == '__main__':
    main()
