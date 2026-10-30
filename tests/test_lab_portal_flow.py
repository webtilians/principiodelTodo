"""Regression and real-process portal lifecycle tests; no production credentials."""
import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from lab_portal.server import app, process_alive, read_job, revision
from lab_portal.catalog import EXPERIMENTS


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, LAB_DATA_DIR=self.tmp.name,
                              LAB_API_TOKEN=secrets.token_urlsafe(32))
        self.env.start()
        self.headers = {'X-Lab-Token': os.environ['LAB_API_TOKEN']}
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.tmp.cleanup()

    def record(self, ident, status='completed', created='2026-01-01'):
        folder = Path(self.tmp.name, ident)
        folder.mkdir()
        (folder / 'job.json').write_text(json.dumps(dict(
            id=ident, experiment='f4', status=status, created_at=created)))
        return folder

    def test_active_job_beyond_first_100_ids_still_blocks_launch(self):
        for i in range(1, 102):
            self.record(f'{i:032x}')
        active = self.record('0' * 32, 'running')
        (active / 'worker.pid').write_text(str(os.getpid()))
        with patch('lab_portal.server.subprocess.Popen') as spawn:
            response = self.client.post('/api/jobs', headers=self.headers,
                                        json={'experiment': 'f4', 'params': {'seeds': 4}})
        self.assertEqual(response.status_code, 409)
        spawn.assert_not_called()
        self.assertEqual(len(self.client.get('/api/jobs', headers=self.headers).json()['jobs']), 102)

    def test_revision_for_deployment_owned_by_another_user(self):
        with patch.dict(os.environ, GIT_TEST_ASSUME_DIFFERENT_OWNER="1"):
            self.assertRegex(revision() or "", r"^[0-9a-f]{40}$")

    def test_history_sorted_by_creation_not_random_uuid(self):
        self.record('f' * 32, created='2026-01-01')
        self.record('0' * 32, created='2026-10-23')
        jobs = self.client.get('/api/jobs', headers=self.headers).json()['jobs']
        self.assertEqual(jobs[0]['id'], '0' * 32)

    def test_spawn_failure_does_not_leave_queue_blocked(self):
        with patch('lab_portal.server.subprocess.Popen', side_effect=OSError('spawn failed')):
            response = self.client.post('/api/jobs', headers=self.headers,
                                        json={'experiment': 'f4', 'params': {'seeds': 4}})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get('/api/jobs', headers=self.headers).json()['jobs'][0]['status'], 'failed')

    def test_stale_queue_without_pid_is_interrupted(self):
        folder = self.record('a' * 32, 'queued')
        self.assertEqual(read_job(folder)['status'], 'queued')
        os.utime(folder / 'job.json', (time.time() - 60,) * 2)
        self.assertEqual(read_job(folder)['status'], 'interrupted')

    def test_liveness_probe_does_not_terminate_process(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            self.assertTrue(process_alive(child.pid))
            time.sleep(.1)
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_real_workers_reports_archives_and_reopened_client(self):
        catalog = self.client.get('/api/catalog', headers=self.headers).json()
        self.assertEqual({x['id'] for x in catalog['experiments']}, set(EXPERIMENTS))
        for experiment in ('f4', 'f5', 'f6', 'f7', 'f7b', 'f8'):
            with self.subTest(experiment=experiment):
                response = self.client.post('/api/jobs', headers=self.headers, json={
                    'experiment': experiment, 'params': EXPERIMENTS[experiment].presets['rapido']})
                self.assertEqual(response.status_code, 201, response.text)
                ident = response.json()['id']
                path = '/api/jobs/' + ident
                deadline = time.monotonic() + 120
                try:
                    while time.monotonic() < deadline:
                        job = self.client.get(path, headers=self.headers).json()
                        if job['status'] not in ('queued', 'running'):
                            break
                        time.sleep(.1)
                    logs = self.client.get(path + '/log', headers=self.headers)
                    self.assertEqual(job['status'], 'completed', (job, logs.text))
                    self.assertEqual(job['exit_code'], 0)
                    self.assertIn('revision', job)
                    report = self.client.get(path + '/report', headers=self.headers)
                    self.assertEqual(report.status_code, 200)
                    self.assertIsInstance(report.json(), dict)
                    summary = self.client.get(path + '/summary', headers=self.headers)
                    self.assertEqual(summary.status_code, 200, summary.text)
                    self.assertTrue(summary.json()['scenarios'])
                    download = self.client.get(path + '/download', headers=self.headers)
                    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
                        self.assertIsNone(archive.testzip())
                        self.assertEqual(json.loads(archive.read('output/report.json')), report.json())
                        self.assertEqual(json.loads(archive.read('job.json'))['id'], ident)
                    self.client.close()
                    self.client = TestClient(app)
                    self.assertEqual(self.client.get(path + '/report', headers=self.headers).content, report.content)
                    # A separate interpreter reads the persisted state, independent of app memory.
                    result = subprocess.check_output([sys.executable, '-c',
                        'import json; from lab_portal.server import list_jobs; print(json.dumps(list_jobs()))'], text=True)
                    self.assertIn(ident, {x['id'] for x in json.loads(result)})
                finally:
                    if self.client.get(path, headers=self.headers).json()['status'] in ('queued', 'running'):
                        self.client.post(path + '/cancel', headers=self.headers)


if __name__ == '__main__':
    unittest.main()
