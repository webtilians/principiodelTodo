"""Actual Windows-agent protocol exercised with real MiniGrid subprocess and TestClient.

Uses in-process authenticated HTTP transport to avoid public credentials.
Requires the F9 pinned heavy dependencies; run only in F9 Actions.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from lab_portal.server import app, CODE_REVISION
from lab_portal.remote_worker import execute_job


class LocalTransport:
    def __init__(self, client, worker_token):
        self.client, self.token = client, worker_token

    def call(self, endpoint, payload=None, *, raw=False, extra_headers=None):
        headers = {"X-Lab-Worker-Token": self.token}
        headers.update(extra_headers or {})
        if raw:
            resp = self.client.post(endpoint, content=payload,
                                    headers={**headers, "Content-Type": "application/zip"})
        else:
            resp = self.client.post(endpoint, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} {resp.text[:300]}")
        return resp.json()


class WindowsEndToEndMiniGrid(unittest.TestCase):
    def test_real_model_trained_uploaded_and_reopened(self):
        with tempfile.TemporaryDirectory() as temporary:
            remote_token = "remote-token-" + "M" * 50
            with patch.dict(os.environ, {
                "LAB_API_TOKEN":"browser-token-for-CI",
                "LAB_WORKER_TOKEN":remote_token,
                "LAB_F9_REMOTE_ENABLED":"1",
                "LAB_F9_ENABLED":"0",
                "LAB_DATA_DIR": temporary + "/vps",
            }):
                with TestClient(app) as client:
                    browser = {"X-Lab-Token":"browser-token-for-CI"}
                    job = client.post("/api/jobs", headers=browser, json={
                        "experiment":"f9_memory",
                        "params":{"seeds":1,"timesteps":128,"eval-episodes":2}})
                    self.assertEqual(job.status_code,201,job.text)
                    id = job.json()["id"]
                    transport = LocalTransport(client, remote_token)
                    ping=transport.call("/api/worker/ping",{
                        "worker_id":"windows-ci","revision":CODE_REVISION})
                    self.assertEqual(ping["jobs"][0]["id"],id)
                    claim=transport.call(f"/api/worker/jobs/{id}/claim",{
                        "worker_id":"windows-ci","revision":CODE_REVISION})
                    result=execute_job(
                        transport,claim,python=sys.executable,max_seconds=120,
                        runs_root=Path(temporary)/"windows")
                    self.assertEqual(result["status"],"completed")
                    saved=client.get(f"/api/jobs/{id}/report",headers=browser)
                    self.assertEqual(saved.status_code,200)
                    report=saved.json()
                    self.assertEqual(report["phase"],"development")
                    self.assertEqual(set(report["summary"]["policies"]),{
                        "random","ppo","recurrent_ppo"})
                    self.assertEqual(report["training_seeds"],1)
                    summary=client.get(f"/api/jobs/{id}/summary",headers=browser)
                    self.assertEqual(summary.status_code,200,summary.text)
                    disk=Path(temporary,"vps",id,"output")
                    self.assertTrue(list((disk/"checkpoints").glob("*.zip")))
                    self.assertEqual(json.loads((disk/"report.json").read_text()),report)


if __name__=="__main__":
    unittest.main()
