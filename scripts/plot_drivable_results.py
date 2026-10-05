"""Plot preserved official-validation scores and the separate development history."""
import argparse
import json
from pathlib import Path


def plot(directory, baseline):
    report = json.loads((directory / 'official_val_result.json').read_text())
    reference = json.loads((baseline / 'result.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    best = json.loads((directory / 'best_selection.json').read_text())
    identities = lambda value: [(row['id'], row['image_sha256'], row['mask_sha256']) for row in value['samples']]
    if identities(report) != identities(reference) or report['label_schema'] != reference['label_schema']:
        raise ValueError('Validation inputs/label policies differ')
    if [row['epoch'] for row in history] != list(range(1, 51)):
        raise ValueError('Incomplete epoch history')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout='constrained')
    positions = np.arange(3)
    for shift, value, colour, name in ((-0.18, reference, '#75869e', 'ADE20K reference'),
                                       (0.18, report, '#087f8c', 'IDD Lite fine-tuned')):
        bars = axes[0].bar(positions + shift, [value['binary_road'][key] * 100 for key in ('IoU', 'precision', 'recall')],
                           width=0.34, color=colour, label=name)
        axes[0].bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    axes[0].set_xticks(positions, ['Drivable IoU', 'Pixel precision', 'Pixel recall'])
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel('Global binary pixel metric (%)')
    axes[0].set_title('Official validation · all 204 images', loc='left', pad=14)
    axes[0].legend(loc='lower right', frameon=False)
    axes[0].grid(axis='y', alpha=0.15)
    axes[0].set_axisbelow(True)
    epochs = [row['epoch'] for row in history]
    values = [row['development_raw']['IoU'] * 100 for row in history]
    axes[1].plot(epochs, values, color='#087f8c', linewidth=2.4)
    axes[1].scatter([best['epoch']], [best['development_raw']['IoU'] * 100], color='#d97926', s=65, zorder=3)
    axes[1].annotate(f"Selected epoch {best['epoch']}\n{best['development_raw']['IoU'] * 100:.2f}% raw dev IoU",
                     (best['epoch'], best['development_raw']['IoU'] * 100), xytext=(0, -45),
                     textcoords='offset points', ha='center', fontsize=10)
    axes[1].set_xlim(1, 50)
    axes[1].set_ylim(85, 96)
    axes[1].set_xlabel('Completed training epoch')
    axes[1].set_ylabel('Internal development raw IoU (%)')
    axes[1].set_title('Checkpoint selection · 190 dev images', loc='left', pad=14)
    axes[1].grid(alpha=0.15)
    fig.suptitle('RoadSense India · SegFormer-B0 drivable-area fine-tuning', fontsize=17)
    gain = (report['binary_road']['IoU'] - reference['binary_road']['IoU']) * 100
    fig.supxlabel(f'{gain:.2f} percentage-point validation IoU gain · 1,213 fit images · 50 epochs · best epoch 36\n'
                  'Validation: deployed mask postprocessing. Development: raw argmax. CPU/GPU experiments differ; not lane/pothole accuracy.', fontsize=10)
    for suffix in ('png', 'svg'):
        path = directory / ('comparison.' + suffix)
        fig.savefig(path, dpi=180)
        if suffix == 'svg':
            # Normalize generated XML formatting, not any measured evidence.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('benchmarks/idd_lite_finetuned_20261004'))
    parser.add_argument('--baseline', type=Path, default=Path('benchmarks/idd_lite_cpu_20261001'))
    args = parser.parse_args()
    plot(args.directory, args.baseline)
