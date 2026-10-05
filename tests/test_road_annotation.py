"""Annotation-tool tests on synthetic fixtures, never real human-review claims."""
import copy
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image
from test_independent_road_test import inputs as source_inputs

from scripts.import_road_annotations import checked_polygon, import_draft, render_mask
from scripts.prepare_independent_road_test import TAGS, digest, freeze, sample
from scripts.serve_road_annotation import load_package, make_handler

inputs = source_inputs  # Reuse the generated-video fixture, not real road labels.


def fixture_draft(package, plan):
    entries = []
    for row in plan['samples']:
        entries.append({'id': row['id'], 'image_sha256': row['image_sha256'],
                        'polygons': [{'label': 1, 'points': [[0, 24], [63, 24], [63, 47], [0, 47]]},
                                     {'label': 0, 'points': [[10, 30], [15, 30], [15, 35], [10, 35]]},
                                     {'label': 255, 'points': [[20, 30], [25, 30], [25, 35], [20, 35]]}],
                        'negative_confirmed': False, 'annotation_complete': True, 'human_reviewed': True,
                        'annotator_alias': 'synthetic_test_annotator', 'reviewer_alias': 'synthetic_test_reviewer',
                        'tags': {tag: 'uncertain' for tag in TAGS}})
    return {'schema_version': 1, 'frame_plan_sha256': digest(package / 'frame_plan.json'), 'samples': entries}


def run_import(inputs, change=None):
    plan = sample(**inputs)
    package = inputs['output']
    draft = fixture_draft(package, plan)
    if change:
        change(draft)
    source = package.parent / 'draft.json'
    source.write_text(json.dumps(draft))
    output = package.parent / 'imported'
    result = import_draft(package / 'frame_plan.json', source, output)
    return package, output, plan, result


def test_import_rasterizes_native_labels_and_freezes_only_explicit_review(inputs):
    original, output, plan, result = run_import(inputs)
    assert not list((original / 'masks').rglob('*.png'))
    assert result['completed_masks'] == result['human_reviewed'] == 6
    assert result['accuracy'] is None
    with Image.open(output / plan['samples'][0]['mask']) as mask:
        pixels = np.array(mask)
        assert mask.mode == 'L' and mask.size == (64, 48)
    assert pixels[0, 0] == 0 and pixels[25, 2] == 1
    assert pixels[32, 12] == 0 and pixels[32, 22] == 255
    frozen = freeze(output / 'frame_plan.json', output / 'annotation_reviews.json', output / 'manifest.json')
    assert len(frozen['samples']) == 6 and frozen['accuracy'] is None
    with pytest.raises(ValueError, match='fresh output'):
        import_draft(original / 'frame_plan.json', output / 'annotation_draft.json', output)


def test_incomplete_frames_do_not_gain_fake_masks_or_reviews(inputs):
    def change(draft):
        e = draft['samples'][0]
        e.update(polygons=[], annotation_complete=False, human_reviewed=False, annotator_alias='', reviewer_alias='')
    _, output, plan, result = run_import(inputs, change)
    assert result['completed_masks'] == result['human_reviewed'] == 5
    assert not (output / plan['samples'][0]['mask']).exists()
    with pytest.raises(ValueError, match='Unreviewed'):
        freeze(output / 'frame_plan.json', output / 'annotation_reviews.json', output / 'manifest.json')


def test_same_person_review_is_disclosed(inputs):
    def change(draft):
        draft['samples'][0]['reviewer_alias'] = draft['samples'][0]['annotator_alias']
    _, _, _, result = run_import(inputs, change)
    assert result['self_reviewed'] == 1


@pytest.mark.parametrize('mutation', ['plan_hash', 'image_hash', 'drop', 'duplicate', 'review_alias', 'review_incomplete', 'tags', 'nonboolean', 'negative_missing', 'negative_road', 'incomplete_bad_polygon'])
def test_import_rejects_bad_drafts_before_creating_output(inputs, mutation):
    plan = sample(**inputs)
    package = inputs['output']
    draft = fixture_draft(package, plan)
    e = draft['samples'][0]
    if mutation == 'plan_hash':
        draft['frame_plan_sha256'] = '0' * 64
    elif mutation == 'image_hash':
        e['image_sha256'] = '0' * 64
    elif mutation == 'drop':
        draft['samples'].pop()
    elif mutation == 'duplicate':
        draft['samples'][1] = copy.deepcopy(e)
    elif mutation == 'review_alias':
        e['reviewer_alias'] = ''
    elif mutation == 'review_incomplete':
        e['annotation_complete'] = False
    elif mutation == 'tags':
        e['tags']['rain'] = 'maybe'
    elif mutation == 'nonboolean':
        e['human_reviewed'] = 1
    elif mutation == 'negative_missing':
        e['polygons'] = []
    elif mutation == 'negative_road':
        e['negative_confirmed'] = True
    else:
        e.update(annotation_complete=False, human_reviewed=False)
        e['polygons'][0]['points'][0] = [-1, 24]
    source = package.parent / 'bad.json'
    source.write_text(json.dumps(draft))
    output = package.parent / 'bad_output'
    with pytest.raises(ValueError):
        import_draft(package / 'frame_plan.json', source, output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['label', 'boolean_label', 'float', 'nan', 'oob', 'short', 'repeated', 'degenerate', 'crossing'])
def test_bad_polygon_geometry_rejected(mutation):
    polygon = {'label': 1, 'points': [[0, 0], [5, 0], [5, 4], [0, 4]]}
    if mutation == 'label':
        polygon['label'] = 2
    elif mutation == 'boolean_label':
        polygon['label'] = True
    elif mutation in ('float', 'nan'):
        polygon['points'][0][0] = .5 if mutation == 'float' else float('nan')
    elif mutation == 'oob':
        polygon['points'][0][0] = 6
    elif mutation == 'short':
        polygon['points'] = [[0, 0], [5, 0]]
    elif mutation == 'repeated':
        polygon['points'].append([0, 0])
    elif mutation == 'degenerate':
        polygon['points'] = [[0, 0], [1, 1], [2, 2]]
    else:
        polygon['points'] = [[0, 0], [5, 4], [5, 0], [0, 3]]
    with pytest.raises(ValueError):
        checked_polygon(polygon, [6, 5])


def test_negative_and_equal_class_counts_are_handled_without_dictionary_collision():
    entry = {'polygons': [], 'negative_confirmed': True}
    assert render_mask(entry, [4, 4])
    entry = {'polygons': [{'label': 1, 'points': [[0, 0], [1, 0], [1, 3], [0, 3]]},
                          {'label': 255, 'points': [[2, 0], [3, 0], [3, 3], [2, 3]]}], 'negative_confirmed': False}
    assert render_mask(entry, [4, 4])  # 8 road and 8 ignored pixels; equal counts must not collapse.
    entry = {'polygons': [{'label': 255, 'points': [[0, 0], [3, 0], [3, 3], [0, 3]]}], 'negative_confirmed': True}
    with pytest.raises(ValueError, match='Entirely ignored'):
        render_mask(entry, [4, 4])


def test_server_is_read_only_and_serves_only_allowlisted_images(inputs):
    sample(**inputs)
    package, plan, public = load_package(inputs['output'])
    assert 'checkpoint_path' not in public and public['frame_plan_sha256'] == digest(package / 'frame_plan.json')
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(package, plan, public))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection('127.0.0.1', server.server_port)
    try:
        for path, expected in [('/', 200), ('/app.js', 200), ('/style.css', 200), ('/plan.json', 200), ('/images/0', 200), ('/images/99', 404), ('/images/../config.yaml', 404), ('/frame_plan.json', 404), ('/best/model.safetensors', 404)]:
            client.request('GET', path)
            response = client.getresponse()
            assert response.status == expected
            if expected == 200:
                assert "frame-ancestors 'none'" in response.getheader('Content-Security-Policy')
            response.read()
        client.request('GET', '/plan.json', headers={'Host': 'untrusted.example'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.request('POST', '/plan.json', body=b'{}')
        response = client.getresponse()
        assert response.status == 501
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_changed_source_cannot_be_served(inputs):
    plan = sample(**inputs)
    (inputs['output'] / plan['samples'][0]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed image'):
        load_package(inputs['output'])
