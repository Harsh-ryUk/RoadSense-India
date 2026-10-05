"""Rasterize explicitly completed human polygon drafts into a NEW review package."""
import argparse
import io
import json
import math
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_independent_road_test import (
    TAG_VALUES,
    TAGS,
    digest,
    frozen_configuration,
    inside,
    validate_mask,
    write_new_json,
)


def checked_polygon(polygon, resolution):
    label, points = polygon.get('label'), polygon.get('points')
    if type(label) is not int or label not in (0, 1, 255):
        raise ValueError('Polygon labels must be integer 0, 1 or 255')
    if not isinstance(points, list) or not 3 <= len(points) <= 200:
        raise ValueError('Each polygon needs 3 to 200 native-image points')
    width, height = resolution
    checked = []
    for point in points:
        if (not isinstance(point, list) or len(point) != 2
                or any(type(n) is not int or not math.isfinite(n) for n in point)
                or not 0 <= point[0] < width or not 0 <= point[1] < height):
            raise ValueError('Polygon points must be in-bounds integer pixel coordinates')
        checked.append(tuple(point))
    if len(set(checked)) != len(checked):
        raise ValueError('Repeated polygon vertices')
    edges = list(zip(checked, checked[1:] + checked[:1]))
    if sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) == 0:
        raise ValueError('Degenerate polygon')

    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return orientation(a, b, c) == 0 and min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            c, d = edges[j]
            crossing = orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0
            if crossing or any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b))):
                raise ValueError('Self-intersecting polygon')
    return label, checked


def render_mask(entry, resolution):
    polygons = entry.get('polygons')
    if not isinstance(polygons, list) or len(polygons) > 100:
        raise ValueError('Expected up to 100 polygons per image')
    if type(entry.get('negative_confirmed')) is not bool:
        raise ValueError('Explicit negative confirmation must be boolean')
    mask = Image.new('L', tuple(resolution), 0)
    painter = ImageDraw.Draw(mask)
    for polygon in polygons:
        label, points = checked_polygon(polygon, resolution)
        painter.polygon(points, fill=label)
    counts = mask.getcolors(maxcolors=256)
    road_pixels = sum(count for count, value in counts if value == 1)
    valid_pixels = sum(count for count, value in counts if value != 255)
    if not valid_pixels:
        raise ValueError('Entirely ignored mask is not allowed')
    if (road_pixels == 0) != entry['negative_confirmed']:
        raise ValueError('A no-road frame needs explicit negative confirmation; a road frame must not claim it')
    buffer = io.BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()


def import_draft(plan_path, annotations_path, output):
    plan_path, annotations_path, output = Path(plan_path).resolve(), Path(annotations_path).resolve(), Path(output).resolve()
    source = plan_path.parent
    if output.exists() or annotations_path.stat().st_size > 20 * 1024 ** 2:
        raise ValueError('Use a fresh output directory and a draft smaller than 20 MiB')
    plan_hash, draft_hash = digest(plan_path), digest(annotations_path)
    plan, draft = json.loads(plan_path.read_text()), json.loads(annotations_path.read_text())
    if plan.get('stage') != 'awaiting_human_annotations' or draft.get('schema_version') != 1 or draft.get('frame_plan_sha256') != plan_hash:
        raise ValueError('Draft must match the original pending frame plan')
    if frozen_configuration(source / 'evaluation_config.yaml', plan['checkpoint_path']) != plan['evaluation_contract']:
        raise ValueError('Frozen model/configuration changed')
    if digest(Path(plan['training_manifest_path'])) != plan['training_manifest_sha256']:
        raise ValueError('Training exclusion reference changed')
    rows = plan['samples']
    entries = draft.get('samples', [])
    by_id = {entry['id']: entry for entry in entries}
    if len(by_id) != len(entries) or set(by_id) != {row['id'] for row in rows}:
        raise ValueError('Draft must retain every frame exactly once')
    rendered, reviews = {}, []
    for row in rows:
        image = inside(source, row['image'])
        entry = by_id[row['id']]
        if digest(image) != row['image_sha256'] or entry.get('image_sha256') != row['image_sha256']:
            raise ValueError('Changed or wrong image: ' + row['id'])
        with Image.open(image) as pixels:
            if pixels.mode != 'RGB' or list(pixels.size) != row['resolution']:
                raise ValueError('Image mode or resolution changed')
        if type(entry.get('annotation_complete')) is not bool or type(entry.get('human_reviewed')) is not bool:
            raise ValueError('Completion/review declarations must be boolean')
        if (type(entry.get('negative_confirmed')) is not bool
                or any(not isinstance(entry.get(key), str) for key in ('annotator_alias', 'reviewer_alias'))
                or not isinstance(entry.get('polygons'), list) or len(entry['polygons']) > 100):
            raise ValueError('Invalid draft fields')
        for polygon in entry['polygons']:
            checked_polygon(polygon, row['resolution'])
        tags = entry.get('tags', {})
        if set(tags) != set(TAGS) or any(tag not in TAG_VALUES for tag in tags.values()):
            raise ValueError('All scenario tags must be present, absent or uncertain')
        reviewed = entry['human_reviewed']
        if reviewed and (not entry['annotation_complete'] or any(not isinstance(entry.get(key), str) or not entry[key].strip() for key in ('annotator_alias', 'reviewer_alias'))):
            raise ValueError('Review needs completed annotation and actual annotator/reviewer aliases')
        if entry['annotation_complete']:
            if not isinstance(entry.get('annotator_alias'), str) or not entry['annotator_alias'].strip():
                raise ValueError('Completed annotation needs an actual annotator alias')
            rendered[row['id']] = render_mask(entry, row['resolution'])
        review = {'id': row['id'], 'image_sha256': row['image_sha256'], 'human_reviewed': reviewed,
                  'annotator_alias': entry.get('annotator_alias', ''), 'reviewer_alias': entry.get('reviewer_alias', ''),
                  'mask_sha256': '', 'tags': tags}
        reviews.append(review)
    if digest(plan_path) != plan_hash or digest(annotations_path) != draft_hash:
        raise ValueError('Draft or plan changed during validation')
    output.mkdir(parents=True, exist_ok=False)
    for row, review in zip(rows, reviews):
        image, mask = output / row['image'], output / row['mask']
        image.resolve().relative_to(output)
        mask.resolve().relative_to(output)
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(source, row['image']), image)
        if digest(image) != row['image_sha256']:
            raise ValueError('Copied image changed')
        if row['id'] in rendered:
            with mask.open('xb') as handle:
                handle.write(rendered[row['id']])
            validate_mask(mask, row['resolution'])
            review['mask_sha256'] = digest(mask)
    shutil.copy2(plan_path, output / 'frame_plan.json')
    shutil.copy2(source / 'evaluation_config.yaml', output / 'evaluation_config.yaml')
    shutil.copy2(annotations_path, output / 'annotation_draft.json')
    if (digest(output / 'frame_plan.json') != plan_hash or digest(output / 'annotation_draft.json') != draft_hash
            or frozen_configuration(output / 'evaluation_config.yaml', plan['checkpoint_path']) != plan['evaluation_contract']):
        raise ValueError('Plan, draft or evaluation contract changed during import')
    write_new_json(output / 'annotation_reviews.json', {'schema_version': 1, 'samples': reviews})
    result = {'stage': 'awaiting_human_annotations', 'frames': len(rows), 'completed_masks': len(rendered),
              'human_reviewed': sum(r['human_reviewed'] for r in reviews),
              'self_reviewed': sum(r['human_reviewed'] and r['annotator_alias'] == r['reviewer_alias'] for r in reviews),
              'source_plan_sha256': plan_hash, 'draft_sha256': draft_hash, 'accuracy': None,
              'rasterization': 'Pillow filled polygons at integer native pixel coordinates; ordered polygons, last wins; no antialiasing.',
              'limitation': 'Human declarations are recorded, not independently certified; polygon quality still needs review.'}
    write_new_json(output / 'annotation_import.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'annotations', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(import_draft(args.plan, args.annotations, args.output), indent=2))


if __name__ == '__main__':
    main()
