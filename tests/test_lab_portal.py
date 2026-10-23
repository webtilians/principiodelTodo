"""INFINITO portal: API authorization, allowlisting, job lifecycle, provenance."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from lab_portal.catalog import EXPERIMENTS, ParameterError, command, normalize
from lab_portal.server import app
from lab_portal.summaries import summarize


class CatalogTests(unittest.TestCase):
    def test_all_experiments_have_valid_presets(self):
        self.assertEqual(set(EXPERIMENTS), {"f4", "f5", "f6", "f7", "f7b"})
        for key, exp in EXPERIMENTS.items():
            for preset in exp.presets.values():
                normalize(key, preset)

    def test_no_shell_args_possible(self):
        for raw in ({"seeds": "; rm -rf /"}, {"seeds": True},
                    {"seeds": 24, "--output": "/tmp/hack"},
                    {"seeds": 100000000}):
            with self.assertRaises(ParameterError):
                normalize("f4", raw)

    def test_command_matches_exact_public_arg_names(self):
        params = normalize("f7", {"seeds": 4, "train": 40, "test": 100})
        args = command("f7", params, "/safe/output")
        self.assertEqual(args[:2], ["-m", "observatory.active_information_f7"])
        self.assertEqual(args[-2:], ["--output", "/safe/output"])
        self.assertNotIn("shell", str(args))

    def test_test_cases_need_even_length(self):
        with self.assertRaises(ParameterError):
            normalize("f7b", {"test": 101})


class APIConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"LAB_API_TOKEN": "unit-test-super-secret", "LAB_DATA_DIR": self.tmp.name})
        self.env.start()
        self.client = TestClient(app)
        self.headers = {"X-Lab-Token": "unit-test-super-secret"}

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.tmp.cleanup()

    def test_unauthenticated_requests_fail(self):
        for path in ("/api/jobs", "/api/catalog", "/api/ping"):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post("/api/jobs",json={"experiment":"f4"}).status_code, 401)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_catalog_lists_only_implemented_protocols(self):
        data = self.client.get("/api/catalog", headers=self.headers).json()
        self.assertEqual(len(data["experiments"]), 5)
        self.assertEqual(data["max_concurrent"], 1)

    def test_create_job_uses_detached_allowlisted_worker(self):
        with patch("lab_portal.server.subprocess.Popen") as popen:
            popen.return_value.pid = 15123
            reply = self.client.post("/api/jobs", headers=self.headers,
                                     json={"experiment":"f4","params":{"seeds":4}})
        self.assertEqual(reply.status_code, 201, reply.text)
        job = reply.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(len(job["id"]), 32)
        record = Path(self.tmp.name, job["id"], "job.json")
        self.assertTrue(record.is_file())
        self.assertEqual(json.loads(record.read_text())["params"], {"seeds":4})
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}", headers=self.headers).status_code,200)
        self.assertIn("lab_portal.worker", str(popen.call_args))

    def test_reject_invalid_ids_and_field_injection(self):
        self.assertEqual(self.client.get("/api/jobs/../../etc/passwd",headers=self.headers).status_code,404)
        response = self.client.post("/api/jobs", headers=self.headers,
                                    json={"experiment":"f6","params":{"seeds":8},"command":"bash"})
        self.assertEqual(response.status_code,422)
        response = self.client.post("/api/jobs", headers=self.headers,
                                    json={"experiment":"f8","params":{}})
        self.assertEqual(response.status_code,422)

    def test_report_and_summary_only_from_own_job(self):
        ident="a"*32
        folder=Path(self.tmp.name,ident,"output")
        folder.mkdir(parents=True)
        (folder.parent / "job.json").write_text(json.dumps({"id":ident,"status":"completed","experiment":"f4"}))
        data={"conditions":{"drifting_relevance":{"strategies":{"adaptive":{"heldout_hit":.88},"fifo":{"heldout_hit":.38}}}}}
        (folder / "report.json").write_text(json.dumps(data))
        response=self.client.get(f"/api/jobs/{ident}/summary",headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["scenarios"]["drifting_relevance"]["adaptive"],.88)
        zipfile=self.client.get(f"/api/jobs/{ident}/download",headers=self.headers)
        self.assertEqual(zipfile.status_code,200)
        self.assertTrue(zipfile.content.startswith(b"PK"))

    def test_prevents_parallel_work(self):
        ident="b"*32
        folder=Path(self.tmp.name,ident)
        folder.mkdir()
        (folder / "job.json").write_text(json.dumps({"id":ident,"status":"running"}))
        (folder / "worker.pid").write_text(str(os.getpid()))
        result=self.client.post("/api/jobs",headers=self.headers,json={"experiment":"f4","params":{"seeds":4}})
        self.assertEqual(result.status_code,409)


class SummaryTests(unittest.TestCase):
    def test_summary_f7b_presentable(self):
        report={"conditions":{"switch":{"mean_reward":{"all":.32,"after_last_100":.35}}}}
        row=summarize("f7b",report)
        self.assertEqual(row["scenarios"]["switch"]["recalibration"],.32)

    def test_f5_does_not_plot_validation_boolean(self):
        report={"conditions":{"stable":{"policies":{"fifo":{"mse":1.23}}},
                              "predeclared_success":{"ok":False}}}
        row=summarize("f5",report)
        self.assertEqual(set(row["scenarios"]),{"stable"})


if __name__ == "__main__":
    unittest.main()
