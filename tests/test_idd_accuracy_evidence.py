"""Audit the real IDD Lite baseline without weights, images or network access."""
import copy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.audit_accuracy_report import audit


EVIDENCE = Path(__file__).resolve().parents[1] / 'benchmarks/idd_lite_cpu_20261001'


def load():
    return (json.loads((EVIDENCE / 'result.json').read_text()),
            json.loads((EVIDENCE / 'sample_index.json').read_text()))


def test_all_official_validation_pairs_and_counts_are_retained():
    report, manifest = load()
    result = audit(report, manifest)
    assert result['images'] == 204
    assert result['drives'] == 61
    assert result['road_status'] == {'observed': 200, 'unknown': 4}
    assert report['selection']['policy'] == 'all validation images'
    assert report['selection']['available_images'] == report['selection']['selected_images'] == 204
    assert report['environment']['device'] == 'cpu'
    assert report['environment']['gpu'] is None
    assert report['detection'] is None
    assert report['label_schema']['road_ids'] == [0]
    assert report['label_schema']['ignore_ids'] == [255]
    assert report['binary_road']['IoU'] == pytest.approx(0.7986843482564515)
    assert all(row['resolution'] == [320, 227] for row in report['samples'])


@pytest.mark.parametrize('mutation', ['metric', 'pixels', 'hash', 'duplicate', 'status', 'negative', 'empty'])
def test_audit_rejects_corrupt_or_incomplete_evidence(mutation):
    report, manifest = load()
    report = copy.deepcopy(report)
    if mutation == 'metric':
        report['binary_road']['IoU'] += 0.01
    elif mutation == 'pixels':
        report['samples'][0]['fn'] += 1
    elif mutation == 'hash':
        report['samples'][0]['mask_sha256'] = 'changed'
    elif mutation == 'duplicate':
        report['samples'][1]['id'] = report['samples'][0]['id']
    elif mutation == 'status':
        report['road_observation_frames']['unknown'] += 1
    elif mutation == 'negative':
        report['samples'][0]['fp'] = -1
    else:
        report['samples'] = []
    with pytest.raises(ValueError):
        audit(report, manifest)


def test_frozen_executable_source_and_raw_report_hashes():
    report, _ = load()
    provenance = json.loads((EVIDENCE / 'provenance.json').read_text())
    for filename, expected in provenance['artifact_sha256'].items():
        assert hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest() == expected, filename
    with tarfile.open(EVIDENCE / 'measured_source.tar.gz') as archive:
        for path, expected in report['source_files_sha256'].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected, path
    assert provenance['source_commit'] == '1ccec21bfa2c2515b16ecc3ae27ff84761e91fa1'
