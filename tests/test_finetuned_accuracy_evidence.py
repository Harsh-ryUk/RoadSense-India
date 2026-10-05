"""Verify completed training/validation artifacts without licensed data or weights."""
import copy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.audit_accuracy_report import audit

EVIDENCE = Path(__file__).resolve().parents[1] / 'benchmarks/idd_lite_finetuned_20261004'
BASELINE = EVIDENCE.parent / 'idd_lite_cpu_20261001'


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def test_complete_official_validation_and_exact_pixel_accounting():
    report, index = load('official_val_result.json'), load('sample_index.json')
    result = audit(report, index)
    assert result == load('audit.json')
    assert result['images'] == 204 and result['drives'] == 61
    assert result['road_status'] == {'observed': 204}
    assert report['selection']['available_images'] == report['selection']['selected_images'] == 204
    assert report['environment']['gpu'] == 'Tesla T4'
    assert report['detection'] is None
    assert result['counts'] == {'tp': 4451862, 'fp': 115523, 'fn': 258081, 'tn': 9990043,
                                'valid_pixels': 14815509, 'ignored_pixels': 3051}
    assert result['metrics']['IoU'] == pytest.approx(0.92257659674734)
    assert result['metrics']['precision'] == pytest.approx(0.9747069712756862)
    assert result['metrics']['recall'] == pytest.approx(0.945205069360712)


def test_preserved_baseline_has_identical_inputs_and_label_policy():
    baseline = json.loads((BASELINE / 'result.json').read_text())
    current = load('official_val_result.json')
    identities = lambda report: [(row['id'], row['image_sha256'], row['mask_sha256']) for row in report['samples']]
    assert identities(current) == identities(baseline)
    assert current['label_schema'] == baseline['label_schema']
    assert 100 * (current['binary_road']['IoU'] - baseline['binary_road']['IoU']) == pytest.approx(12.38922484908885)


def test_all_epochs_selection_numerics_and_checkpoint_identity_agree():
    history, best, summary, protocol = (load(name) for name in
        ('history.json', 'best_selection.json', 'training_summary.json', 'protocol.json'))
    assert [row['epoch'] for row in history] == list(range(1, 51))
    assert summary['training_completed'] and summary['completed_epochs'] == 50
    assert best == history[35] and best['epoch'] == 36
    assert max(row['development_raw']['IoU'] for row in history) == best['development_raw']['IoU']
    assert summary['official_val_metrics'] is None
    assert sum(row['optimizer_updates'] for row in history) == summary['optimizer_updates_total'] == 7595
    assert sum(row['amp_skipped_updates'] for row in history) == summary['amp_skipped_updates_total'] == 5
    assert not protocol['official_val_used_for_selection'] and not protocol['official_val_inference_during_training']
    assert protocol['counts'] == {'fit': {'images': 1213, 'drives': 263},
                                  'development': {'images': 190, 'drives': 46},
                                  'official_val': {'images': 204, 'drives': 61}}
    assert summary['best_weights_sha256'] == load('provenance.json')['selected_weights_sha256']
    assert summary['best_weights_sha256'] == load('official_val_result.json')['models']['segformer_local_files_sha256']['model.safetensors']
    assert load('model_config.json')['id2label'] == {'0': 'non_drivable', '1': 'drivable'}
    assert load('cuda_regression.json')['passed']
    assert not load('targets_result.json')['all_met']


def test_split_folders_and_exact_image_identities_are_disjoint():
    manifest = load('training_splits.json')
    from scripts.prepare_idd_lite_training import check_separation
    check_separation(manifest['splits'])
    assert manifest['counts'] == load('protocol.json')['counts']
    assert hashlib.sha256((EVIDENCE / 'training_splits.json').read_bytes()).hexdigest() == load('protocol.json')['manifest_sha256']


def test_artifact_and_exact_measured_source_hashes():
    provenance = load('provenance.json')
    for name, expected in provenance['artifact_sha256'].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected, name
    expected = dict(load('official_val_result.json')['source_files_sha256'])
    expected.update(load('protocol.json')['source_sha256'])
    with tarfile.open(EVIDENCE / 'measured_source.tar.gz') as archive:
        names = archive.getnames()
        assert all(member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts for member in archive)
        assert all(Path(name).suffix in ('.py', '.txt') for name in names)
        for name, digest in expected.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    assert provenance['independent_dashcam_test_accuracy'] is None
    assert provenance['full_pipeline_latency_with_finetuned_checkpoint'] is None


def test_portable_manifest_only_changes_dataset_root_paths():
    raw, portable = load('official_val_manifest.json'), load('sample_index.json')
    original = copy.deepcopy(raw)
    for row in original['samples']:
        for field in ('image', 'mask'):
            row[field] = str(Path(row[field]).relative_to('/content/idd_data/idd20k_lite'))
    assert original == portable
    assert hashlib.sha256((EVIDENCE / 'official_val_manifest.json').read_bytes()).hexdigest() == load('official_val_result.json')['manifest_sha256']
