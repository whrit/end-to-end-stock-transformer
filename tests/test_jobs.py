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

    def test_slots_run_jobs_side_by_side_and_exclusive_jobs_run_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            store = jobs.JobStore(directory)
            store.put_json("j1/job.json", {"id": "j1"})
            store.put_json("j2/job.json", {"id": "j2", "exclusive": True})
            store.put_json("j3/job.json", {"id": "j3"})
            with mock.patch.object(jobs, "gpu_name", return_value="test GPU"):
                agent = jobs.Agent(store, "a", Path(directory) / "work", slots=2)
            release = threading.Event()

            def fake_run(job_id, job):
                release.wait(10)
                with agent.lock:
                    agent.running.pop(job_id)

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


if __name__ == "__main__":
    unittest.main()
