"""End-to-end tests for the authenticated Windows pull queue, no PyTorch needed."""
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from lab_portal.server import app, CODE_REVISION
from lab_portal.catalog import EXPERIMENTS
from lab_portal.remote_api import validate_bundle
from lab_portal.remote_worker import server_url


class RemoteWorkerFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.token = "worker-shared-secret-from-vps-" + "X" * 50
        self.env = patch.dict(os.environ, {
            "LAB_DATA_DIR": self.tmp.name,
            "LAB_API_TOKEN": "browser-super-secret",
            "LAB_WORKER_TOKEN": self.token,
            "LAB_F9_REMOTE_ENABLED": "1",
            "LAB_F9_ENABLED": "0",
        })
        self.env.start()
        self.client = TestClient(app)
        self.browser = {"X-Lab-Token": "browser-super-secret"}
        self.worker = {"X-Lab-Worker-Token": self.token}

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.tmp.cleanup()

    def post_browser(self, exp="f9_memory"):
        return self.client.post("/api/jobs", headers=self.browser, json={
            "experiment": exp, "params": EXPERIMENTS[exp].presets["rapido"]})

    def ping(self, sha=CODE_REVISION):
        return self.client.post("/api/worker/ping", headers=self.worker, json={
            "worker_id": "enrique-windows", "revision": sha})

    def claim(self, job_id, sha=CODE_REVISION):
        return self.client.post(
            f"/api/worker/jobs/{job_id}/claim", headers=self.worker,
            json={"worker_id": "enrique-windows", "revision": sha})

    @staticmethod
    def archive(task="memory", *, traversal=False, with_report=True):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED) as z:
            if with_report:
                z.writestr("report.json", json.dumps({
                    "phase": "development", "task": task,
                    "summary": {"policies": {
                        "ppo": {"success_rate": .25},
                        "recurrent_ppo": {"success_rate": .5}}}}))
            z.writestr("frozen_manifest.json", json.dumps({
                "stage": "development_complete", "models": {}}))
            z.writestr("../escape.txt" if traversal else
                       "checkpoints/memory_ppo_s0.zip", b"PKnot-model-bytes")
            z.writestr("stdout.log", b"Real experiment output\n")
        return b.getvalue()

    def test_remote_queue_claim_upload_and_browser_download(self):
        created = self.post_browser()
        self.assertEqual(created.status_code, 201, created.text)
        job = created.json()
        self.assertEqual(job["executor"], "remote_windows")
        self.assertEqual(job["status"], "queued")
        self.assertNotIn("claim_id", job)
        job_id = job["id"]
        self.assertEqual(self.ping().json()["jobs"][0]["id"], job_id)
        response = self.claim(job_id)
        self.assertEqual(response.status_code, 200, response.text)
        claim = response.json()["claim_id"]
        self.assertEqual(self.claim(job_id).status_code, 409)
        hb = self.client.post(
            f"/api/worker/jobs/{job_id}/heartbeat", headers=self.worker,
            json={"claim_id": claim, "log_tail": "Optimizing\n"})
        self.assertFalse(hb.json()["cancel"])
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}",
                         headers=self.browser).json()["status"], "running")
        upload = self.client.post(
            f"/api/worker/jobs/{job_id}/finish", content=self.archive(),
            headers={**self.worker, "X-Lab-Claim-ID": claim,
                     "X-Lab-Exit-Code": "0", "Content-Type": "application/zip"})
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(upload.json()["status"], "completed")
        report = self.client.get(f"/api/jobs/{job_id}/report", headers=self.browser)
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["task"], "memory")
        summary = self.client.get(f"/api/jobs/{job_id}/summary", headers=self.browser)
        self.assertEqual(summary.json()["scenarios"]["memory"]["recurrent_ppo"], .5)
        download = self.client.get(f"/api/jobs/{job_id}/download", headers=self.browser)
        self.assertEqual(download.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(download.content)) as z:
            self.assertIn("output/checkpoints/memory_ppo_s0.zip", z.namelist())
            self.assertEqual(z.read("output/report.json"), report.content)
            self.assertIn(b"Real experiment", z.read("stdout.log"))
        self.assertEqual(self.ping().json()["jobs"], [])

    def test_bad_worker_token_and_untrusted_code_rejected(self):
        r = self.client.post("/api/worker/ping",
                             headers={"X-Lab-Worker-Token":"bad"},
                             json={"worker_id":"w","revision":CODE_REVISION})
        self.assertEqual(r.status_code, 401)
        job = self.post_browser().json()
        self.assertEqual(self.ping(sha="f"*40).json()["jobs"], [])
        self.assertEqual(self.claim(job["id"], sha="f"*40).status_code, 409)
        self.assertEqual(self.client.post(
            f"/api/worker/jobs/{job['id']}/claim", headers=self.browser,
            json={"worker_id":"w","revision":CODE_REVISION}).status_code, 401)

    def test_traversal_and_wrong_task_cannot_finalize(self):
        job=self.post_browser().json()
        claim=self.claim(job["id"]).json()["claim_id"]
        endpoint=f"/api/worker/jobs/{job['id']}/finish"
        headers={**self.worker,"X-Lab-Claim-ID":claim}
        illegal=self.client.post(endpoint,content=self.archive(traversal=True),
                                 headers=headers)
        self.assertEqual(illegal.status_code,422,illegal.text)
        wrong=self.client.post(endpoint,content=self.archive(task="doorkey"),
                               headers=headers)
        self.assertEqual(wrong.status_code,422,wrong.text)
        self.assertEqual(self.client.get(
            f"/api/jobs/{job['id']}",headers=self.browser).json()["status"], "running")
        # Retrying after malformed and wrong-task uploads must not reuse stale
        # checkpoint/report files from their rejected staging directories.
        valid = self.client.post(endpoint, content=self.archive(task="memory"),
                                 headers=headers)
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(self.client.get(
            f"/api/jobs/{job['id']}/report",
            headers=self.browser).json()["task"], "memory")

    def test_cancel_does_not_touch_vps_process(self):
        job=self.post_browser().json()
        claim=self.claim(job["id"]).json()["claim_id"]
        with patch("lab_portal.server.os.killpg") as kill:
            response=self.client.post(f"/api/jobs/{job['id']}/cancel",headers=self.browser)
        self.assertEqual(response.status_code,200)
        kill.assert_not_called()
        heartbeat=self.client.post(
            f"/api/worker/jobs/{job['id']}/heartbeat",headers=self.worker,
            json={"claim_id":claim})
        self.assertTrue(heartbeat.json()["cancel"])

    def test_worker_disabled_without_admin_configuration(self):
        with patch.dict(os.environ,{"LAB_F9_REMOTE_ENABLED":"0"}):
            self.assertEqual(self.post_browser().status_code,503)
            self.assertEqual(self.ping().status_code,401)

    def test_queued_and_running_do_not_require_server_pid(self):
        job=self.post_browser().json()
        job_file=Path(self.tmp.name,job["id"],"job.json")
        record=json.loads(job_file.read_text())
        record["created_epoch"]=time.time()-1900
        job_file.write_text(json.dumps(record))
        status=self.client.get(f"/api/jobs/{job['id']}",headers=self.browser)
        self.assertEqual(status.json()["status"],"interrupted")

    def test_url_requires_https_without_credentials(self):
        self.assertEqual(server_url("https://lab.example.com/"),
                         "https://lab.example.com")
        for value in ("http://example.com", "https://user:pass@example.com",
                      "https://example.com/#fragment",
                      "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                server_url(value)
        self.assertEqual(server_url("http://127.0.0.1:8123",True),
                         "http://127.0.0.1:8123")


if __name__ == "__main__":
    unittest.main()
