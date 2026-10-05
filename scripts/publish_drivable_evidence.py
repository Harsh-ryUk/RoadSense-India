"""Export verified training metadata and measured source; never publish data or weights."""
import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_accuracy_report import audit


def sha(content):
    return hashlib.sha256(content).hexdigest()


def publish(run, bundle, splits, output):
    run, bundle, splits, output = map(Path, (run, bundle, splits, output))
    filenames = ('official_val_result.json', 'official_val_manifest.json', 'protocol.json',
                 'history.json', 'training_summary.json', 'best_selection.json',
                 'targets_result.json', 'evaluation_config.yaml', 'cuda_regression.json',
                 'numerical_events.jsonl', 'training.log', 'targeted_tests.log')
    if any((output / name).exists() for name in (*filenames, 'provenance.json', 'measured_source.tar.gz')):
        raise ValueError('Refusing to overwrite published measurement artifacts')
    content = {name: (run / name).read_bytes() for name in filenames}
    report, manifest, protocol, history, summary, best = (
        json.loads(content[name]) for name in ('official_val_result.json', 'official_val_manifest.json',
                                             'protocol.json', 'history.json', 'training_summary.json',
                                             'best_selection.json'))
    audited = audit(report, manifest)
    if report['manifest_sha256'] != sha(content['official_val_manifest.json']):
        raise ValueError('Measured manifest hash mismatch')
    if not summary['training_completed'] or [row['epoch'] for row in history] != list(range(1, 51)):
        raise ValueError('A complete 50-epoch history is required')
    if summary['completed_epochs'] != len(history) or best != history[best['epoch'] - 1]:
        raise ValueError('Training selection metadata does not agree with the history')
    if protocol['official_val_used_for_selection'] or protocol['official_val_inference_during_training']:
        raise ValueError('Official validation was used during training or selection')
    split_bytes = splits.read_bytes()
    if sha(split_bytes) != protocol['manifest_sha256']:
        raise ValueError('Training split identity mismatch')
    weights = (run / 'best/model.safetensors').read_bytes()
    weight_sha = sha(weights)
    if weight_sha != summary['best_weights_sha256'] or weight_sha != report['models']['segformer_local_files_sha256']['model.safetensors']:
        raise ValueError('Selected/evaluated checkpoint identity mismatch')
    content['training_splits.json'] = split_bytes
    for name in ('config.json', 'preprocessor_config.json'):
        data = (run / 'best' / name).read_bytes()
        if sha(data) != report['models']['segformer_local_files_sha256'][name]:
            raise ValueError('Changed model metadata')
        content['model_' + name] = data
    portable = json.loads(content['official_val_manifest.json'])
    data_root = PurePosixPath(protocol['settings']['root'])
    for row in portable['samples']:
        for field in ('image', 'mask'):
            row[field] = str(PurePosixPath(row[field]).relative_to(data_root))
    content['sample_index.json'] = (json.dumps(portable, indent=2) + '\n').encode()
    content['audit.json'] = (json.dumps(audited, indent=2) + '\n').encode()
    expected = dict(report['source_files_sha256'])
    for name, value in protocol['source_sha256'].items():
        if name in expected and expected[name] != value:
            raise ValueError('Training/evaluation sources disagree: ' + name)
        expected[name] = value
    sources = {}
    with tarfile.open(bundle) as archive:
        for name in expected:
            if name == 'scripts/train_drivable_idd_lite.py':
                data = (run / 'trained_source.py').read_bytes()
            else:
                member = archive.getmember(name)
                if not member.isfile():
                    raise ValueError('Source must be a regular file: ' + name)
                data = archive.extractfile(member).read()
            if sha(data) != expected[name]:
                raise ValueError('Measured source mismatch: ' + name)
            sources[name] = data
        for name in ('scripts/benchmark_pipeline.py', 'requirements.txt', 'requirements-training.txt',
                     'requirements-eval.txt', 'scripts/prepare_idd_lite_eval.py'):
            sources[name] = archive.extractfile(name).read()
    source_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=source_buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, data in sorted(sources.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(data), 0o644, 0
                archive.addfile(member, io.BytesIO(data))
    content['measured_source.tar.gz'] = source_buffer.getvalue()
    provenance = {
        'schema_version': 1, 'measurement_date_utc': report['timestamp_utc'],
        'scope': 'Saved complete training and official-validation evidence; no new inference on publication',
        'original_result_unchanged': True, 'original_manifest_unchanged': True,
        'portable_index_transformation': 'Only image/mask paths changed from the recorded Colab dataset root to dataset-relative paths; IDs, hashes and labels unchanged',
        'selected_weights_sha256': weight_sha,
        'original_training_bundle_sha256': sha(bundle.read_bytes()),
        'source_snapshot': 'Measured evaluator/source from original source bundle; trainer replaced by the exact executed FP32-loss/AMP-stability version, all recorded Python hashes verified',
        'published_source_sha256': {name: sha(data) for name, data in sorted(sources.items())},
        'artifact_sha256': {name: sha(data) for name, data in sorted(content.items())},
        'excluded': ['dataset images', 'annotation masks', 'model weights', 'credentials', 'personal notebook outputs'],
        'independent_dashcam_test_accuracy': None,
        'full_pipeline_latency_with_finetuned_checkpoint': None,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        (output / name).write_bytes(data)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'published': str(output), 'completed_epochs': len(history), 'best_epoch': best['epoch'],
                      'official_validation': audited, 'weights_published': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    publish(args.run, args.bundle, args.splits, args.output)
