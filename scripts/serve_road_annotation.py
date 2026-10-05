"""Read-only loopback annotation UI. Exports drafts; never serves model predictions."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_independent_road_test import (
    TAGS,
    digest,
    frozen_configuration,
    inside,
)

ASSETS = Path(__file__).resolve().parents[1] / 'tools/road_annotation'


def load_package(package):
    package = Path(package).resolve()
    plan_path = package / 'frame_plan.json'
    plan = json.loads(plan_path.read_text())
    if plan.get('stage') != 'awaiting_human_annotations' or not plan.get('samples'):
        raise ValueError('Use an actual model-bound package awaiting human annotations')
    if frozen_configuration(package / 'evaluation_config.yaml', plan['checkpoint_path']) != plan['evaluation_contract']:
        raise ValueError('Checkpoint or configuration changed')
    if digest(Path(plan['training_manifest_path'])) != plan['training_manifest_sha256']:
        raise ValueError('Training reference changed')
    for row in plan['samples']:
        if digest(inside(package, row['image'])) != row['image_sha256']:
            raise ValueError('Changed image: ' + row['id'])
    public = {'schema_version': 1, 'frame_plan_sha256': digest(plan_path), 'tags': list(TAGS),
              'samples': [{key: row[key] for key in ('id', 'image_sha256', 'resolution', 'nominal_time_seconds')}
                          for row in plan['samples']]}
    return package, plan, public


def make_handler(package, plan, public):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                self.send_error(403, 'Use the displayed loopback IP URL')
                return
            route = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                      '/style.css': ('style.css', 'text/css')}
            if route in assets:
                filename, kind = assets[route]
                content = (ASSETS / filename).read_bytes()
            elif route == '/plan.json':
                content, kind = json.dumps(public).encode(), 'application/json'
            elif route.startswith('/images/'):
                index = route.removeprefix('/images/')
                if not index.isascii() or not index.isdigit() or len(index) > 8 or int(index) >= len(plan['samples']):
                    self.send_error(404)
                    return
                row = plan['samples'][int(index)]
                path = inside(package, row['image'])
                if digest(path) != row['image_sha256']:
                    self.send_error(409, 'Image changed after loading')
                    return
                content, kind = path.read_bytes(), 'image/png'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged port from 1024 to 65535')
    package, plan, public = load_package(args.package)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(package, plan, public))
    print(f'Annotation UI: http://127.0.0.1:{server.server_port}/', flush=True)
    print('Read-only server. No prediction/label generation; export drafts to Downloads.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
