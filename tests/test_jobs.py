import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from market_guided_stock_alpha import jobs


class JobQueueTest(unittest.TestCase):
    def test_only_one_agent_claims_a_job_and_skips_cancelled_or_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            store.put_json("j1/job.json", {"id": "j1"})
            store.put_text("j2/code.tar.gz", "partial upload, no job.json yet")
            store.put_json("j3/job.json", {"id": "j3"})
            store.put_json("j3/claim.json", {"cancelled": True})
            store.put_json("_agents/a.json", {"name": "a"})

            ran = []
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"):
                first = jobs.Agent(store, "a", Path(directory) / "work")
                second = jobs.Agent(store, "b", Path(directory) / "work")
            for agent in (first, second):
                agent.run_job = lambda job_id, job, agent=agent: ran.append((agent.name, job_id))

            self.assertTrue(first.poll_once())
            self.assertFalse(second.poll_once())
            self.assertEqual(ran, [("a", "j1")])
            self.assertFalse(store.put_json("j1/claim.json", {"agent": "b"}, exclusive=True))
            self.assertEqual(store.get_json("j1/claim.json")["agent"], "a")

    def test_failed_setup_still_records_state_and_frees_the_job_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            store.put_json("j1/job.json", {"id": "j1"})  # no code.tar.gz: setup fails
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"):
                agent = jobs.Agent(store, "a", Path(directory) / "work")
            with mock.patch.object(jobs.Agent, "heartbeat"):
                self.assertTrue(agent.poll_once())
            self.assertEqual(store.get_json("j1/state.json")["state"], "failed")
            self.assertFalse((Path(directory) / "work" / "j1").exists())
            self.assertEqual((agent.state, agent.job), ("idle", None))

    def test_startup_salvages_runs_of_jobs_the_machine_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory + "/queue")
            runs = Path(directory) / "runs" / "exp"
            store.put_json("j1/job.json", {"id": "j1", "experiment": "exp", "run_location": str(runs)})
            store.put_json("j1/state.json", {"state": "running"})
            store.put_json("j2/job.json", {"id": "j2", "experiment": "done_exp", "run_location": str(runs) + "_done"})
            store.put_json("j2/state.json", {"state": "done"})
            work = Path(directory) / "work"
            trained = work / "j1" / "tmp" / "tmpab12" / "output" / "exp"
            (trained / "seed_42" / "target_2012H1").mkdir(parents=True)
            (trained / "config.json").write_text("{}")
            (trained / "seed_42" / "target_2012H1" / "metrics.json").write_text("{}")
            (work / "j2" / "tmp" / "x" / "output" / "done_exp").mkdir(parents=True)
            (work / "j2" / "tmp" / "x" / "output" / "done_exp" / "config.json").write_text("{}")
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"):
                agent = jobs.Agent(store, "a", work)
            agent.salvage()
            self.assertTrue((runs / "seed_42" / "target_2012H1" / "metrics.json").is_file())
            state = store.get_json("j1/state.json")
            self.assertEqual((state["state"], state["salvaged"]), ("stopped", True))
            self.assertFalse(Path(str(runs) + "_done").exists())  # finished jobs already published
            self.assertEqual(store.get_json("j2/state.json")["state"], "done")

    def test_r2_blip_during_a_job_neither_fails_nor_orphans_it(self):
        import sys

        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            store.put_json("j1/job.json", {"id": "j1"})
            store.put_text("j1/config.json", "{}")
            work = Path(directory) / "work"
            (work / "src").mkdir(parents=True)
            import tarfile
            with tarfile.open(str(Path(directory) / "j1" / "code.tar.gz"), "w:gz") as archive:
                archive.add(str(work / "src"), arcname="src")
            script = "import os, time; print('TMPDIR=' + os.environ['TMPDIR'], flush=True); time.sleep(1.5)"
            real_put_file, calls = store.put_file, []

            def flaky_put_file(relative, path):
                calls.append(relative)
                if len(calls) <= 2:
                    raise ConnectionError("Could not connect to the endpoint URL")
                real_put_file(relative, path)

            store.put_file = flaky_put_file
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"):
                agent = jobs.Agent(store, "a", work)
            with mock.patch.object(jobs.Agent, "heartbeat"), \
                    mock.patch.object(jobs.Agent, "_environment_report", return_value=None), \
                    mock.patch.object(jobs, "job_command", return_value=[sys.executable, "-c", script]), \
                    mock.patch.object(jobs, "LOG_UPLOAD_SECONDS", 0.0), \
                    mock.patch.object(jobs, "RETRY_SECONDS", 0.0):
                agent.run_job("j1", {"id": "j1"})
            self.assertGreaterEqual(len(calls), 3)  # uploads kept going after the failures
            self.assertEqual(store.get_json("j1/state.json")["state"], "done")
            self.assertIn("TMPDIR={}".format(work / "j1" / "tmp"), store.get_text("j1/job.log"))
            self.assertFalse((work / "j1").exists())

    def test_default_slots_fit_the_smallest_gpu(self):
        # nvidia-smi MiB: a 24 GB card reports slightly under 24 GiB
        for output, slots in (("24564\n24564\n", 2), ("16303\n", 1), ("11264\n", 1), ("81559\n", 3), ("", 1)):
            with self.subTest(output=output), mock.patch.object(
                jobs.subprocess, "run", return_value=mock.Mock(stdout=output)
            ):
                self.assertEqual(jobs.default_slots(), slots)

    def test_slots_run_jobs_side_by_side_and_exclusive_jobs_run_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            store.put_json("j1/job.json", {"id": "j1"})
            store.put_json("j2/job.json", {"id": "j2", "exclusive": True})
            store.put_json("j3/job.json", {"id": "j3"})
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"), \
                    mock.patch.object(jobs, "gpu_count", return_value=1):
                agent = jobs.Agent(store, "a", Path(directory) / "work", slots=2)
            release = threading.Event()

            def fake_run(job_id, job):
                release.wait(10)
                with agent.lock:
                    agent.running.pop(job_id)
                    agent.devices.pop(job_id)

            agent.run_job = fake_run

            def fill():
                claimed = []
                while agent._has_room() and agent.poll_once(agent._spawn):
                    claimed.append(agent.threads[-1].name)
                return claimed

            # j2 needs the machine to itself, so the second slot goes to j3.
            self.assertEqual(fill(), ["j1", "j3"])
            self.assertEqual(agent.job, "j1, j3")
            release.set()
            for thread in agent.threads:
                thread.join(10)
            agent.threads = [thread for thread in agent.threads if thread.is_alive()]
            release.clear()
            self.assertEqual(fill(), ["j2"])  # alone: the free slot stays empty
            self.assertFalse(agent._has_room())
            release.set()

    def test_jobs_spread_over_gpus_and_reuse_the_freed_one(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            for name in ("j1", "j2", "j3", "j4", "j5"):
                store.put_json("{}/job.json".format(name), {"id": name})
            with mock.patch.object(jobs, "gpu_name", return_value="2× test GPU"), \
                    mock.patch.object(jobs, "gpu_count", return_value=2):
                agent = jobs.Agent(store, "a", Path(directory) / "work", slots=2)
            for _ in range(4):
                self.assertTrue(agent.poll_once(lambda job_id, job: agent.threads.append(job_id)))
            self.assertEqual(agent.devices, {"j1": 0, "j2": 1, "j3": 0, "j4": 1})
            self.assertFalse(agent._has_room())  # 2 slots x 2 GPUs
            agent.threads.remove("j2")
            del agent.devices["j2"]
            agent.running.pop("j2")
            self.assertTrue(agent.poll_once(lambda job_id, job: agent.threads.append(job_id)))
            self.assertEqual(agent.devices["j5"], 1)  # the GPU j2 left


if __name__ == "__main__":
    unittest.main()
