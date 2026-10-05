"""Check portfolio links and clean notebook code cells without executing experiments."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ('README.md', 'docs/PROJECT_SUMMARY.md', 'docs/DRIVABLE_TRAINING.md',
        'docs/INDEPENDENT_ROAD_TEST.md', 'docs/ROAD_ANNOTATION_STUDIO.md',
        'docs/VALIDATION_ROADMAP.md', 'docs/MODEL_CARD.md', 'docs/ACCURACY_EVALUATION.md',
        'docs/COLAB.md', 'benchmarks/idd_lite_finetuned_20261004/README.md',
        'benchmarks/idd_lite_cpu_20261001/README.md', 'benchmarks/colab_t4_video_20261001/README.md')


def check():
    checked_links = 0
    for name in DOCS:
        path = ROOT / name
        text = path.read_text()
        if re.search(r'\b(?:Codex|ChatGPT)\b|AI-assisted', text, re.IGNORECASE):
            raise ValueError('Unexpected assistant-specific documentation: ' + name)
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', text):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            target = unquote(target.split('#')[0])
            if not (path.parent / target).is_file():
                raise ValueError(f'Broken local file link in {name}: {target}')
            checked_links += 1
    cells = 0
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            if cell.get('outputs') or cell.get('execution_count') is not None:
                raise ValueError('Public workflow notebook contains execution output: ' + path.name)
            source = ''.join(cell['source'])
            plain = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('%', '!')))
            ast.parse(plain, filename=str(path))
            cells += 1
    result = {'local_file_links_checked': checked_links, 'workflow_code_cells_syntax_checked': cells,
              'scope': 'Local file links and Python syntax only; no remote-link, GPU execution or Mermaid rendering check'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    check()
