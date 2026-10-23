"""Opt-in HTTP acceptance check. Read LAB_API_TOKEN privately from the environment."""
import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--experiment', default='f4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=180)
    args = parser.parse_args()
    url = args.url.rstrip('/')
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error('Use a base URL without credentials, query or fragment')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')):
        parser.error('HTTPS required except on localhost')
    token = os.environ.get('LAB_API_TOKEN')
    if not token:
        parser.error('LAB_API_TOKEN is required in the environment; never pass it on the command line')

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # Never forward the credential to another origin.

    opener = urllib.request.build_opener(NoRedirect)

    def request(path, body=None):
        headers = {'X-Lab-Token': token}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(url + '/api/' + path, data=data, headers=headers)
        with opener.open(req, timeout=20) as response:
            return response.read()

    def get(path, body=None):
        return json.loads(request(path, body))

    args.output.mkdir(parents=True, exist_ok=True)
    ping = get('ping')
    catalog = get('catalog')
    exp = next(x for x in catalog['experiments'] if x['id'] == args.experiment)
    active = [j for j in get('jobs')['jobs'] if j['status'] in ('queued', 'running')]
    if active:
        raise SystemExit('An experiment is active; no new job submitted')
    job = get('jobs', {'experiment': exp['id'], 'params': exp['presets']['rapido']})
    ident = job['id']
    print('Submitted:', ident, flush=True)
    # Persist the ID immediately, including if monitoring is interrupted.
    (args.output / 'job-id.txt').write_text(ident, encoding='utf8')
    path = 'jobs/' + ident
    deadline = time.monotonic() + args.wait_seconds
    states = []
    while time.monotonic() < deadline:
        job = get(path)
        if not states or states[-1] != job['status']:
            states.append(job['status'])
        if job['status'] not in ('queued', 'running'):
            break
        time.sleep(.25)
    (args.output / 'stdout.log').write_text(get(path + '/log')['log'], encoding='utf8')
    if job['status'] != 'completed':
        raise SystemExit(f"Job {ident}: {job['status']}; inspect it in the portal (not automatically cancelled)")
    report = request(path + '/report')
    summary = get(path + '/summary')
    archive = request(path + '/download')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.testzip() is None
        assert json.loads(z.read('output/report.json')) == json.loads(report)
        assert json.loads(z.read('job.json'))['id'] == ident
    assert summary['scenarios']
    assert request(path + '/report') == report
    (args.output / 'report.json').write_bytes(report)
    (args.output / 'results.zip').write_bytes(archive)
    evidence = dict(url=url, id=ident, experiment=exp['id'], states=states,
                    revision=ping.get('revision'), job=job, summary=summary,
                    report_sha256=hashlib.sha256(report).hexdigest(),
                    note='HTTP lifecycle passed; server restart/deploy persistence requires separate verification')
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
    print('HTTP_FLOW_OK:', ident)


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f'HTTP {error.code}; no credential or response body logged')
