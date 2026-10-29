"""In-process job queue: one worker thread analyses files sequentially (the models are CPU-bound)."""
import csv
import io
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid

from pipeline import compare
from pipeline.audio import AudioError

from .batch import BatchPlan

JOBS_DIR = os.environ.get("VA_JOBS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "jobs"))
JOB_TTL_SECONDS = int(os.environ.get("VA_JOB_TTL_HOURS", "24")) * 3600


class Job:
    def __init__(self, plan: BatchPlan, upload_dir: str, source: str):
        self.id = uuid.uuid4().hex[:12]
        self.plan = plan
        self.upload_dir = upload_dir
        self.source = source
        self.created = time.time()
        self.started = None
        self.finished = None
        self.state = "queued"  # queued | running | done | failed
        self.current = None
        self.error = None
        self.results: dict[str, dict] = {}
        self.audio_seconds = 0.0
        self.lock = threading.Lock()

    # ---- views -------------------------------------------------------------
    def counts(self):
        c = {"total": len(self.plan.items), "pending": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
        for it in self.plan.items:
            c[it.status] += 1
        return c

    def status(self):
        with self.lock:
            c = self.counts()
            processed = [r for r in self.results.values() if r.get("elapsed")]
            eta = None
            if self.state == "running" and processed:
                avg = sum(r["elapsed"] for r in processed) / len(processed)
                eta = round(avg * (c["pending"] + c["running"]))
            return {
                "id": self.id, "state": self.state, "source": self.source, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished,
                "current": self.current, "counts": c, "eta_seconds": eta,
                "audio_seconds": round(self.audio_seconds, 1),
                "elapsed_seconds": round((self.finished or time.time()) - self.started, 1) if self.started else None,
                "validation": {k: v for k, v in self.plan.to_dict().items() if k != "items"},
                "files": [self._file_view(it) for it in self.plan.items],
                "comparison": self.comparison() if self.plan.has_labels else None,
            }

    def _file_view(self, it):
        r = self.results.get(it.name, {})
        return {
            "name": it.name, "status": it.status, "error": it.error, "warnings": it.warnings,
            "in_manifest": it.in_manifest, "has_label": bool(it.label),
            "result": r.get("result"), "reasons": r.get("reasons"), "duration": r.get("duration"),
            "elapsed": r.get("elapsed"),
            "matches": compare.field_matches(r["result"], it.label) if (r.get("result") and it.label) else None,
            "label": it.label,
        }

    def comparison(self):
        pairs = [(self.results[it.name]["result"], it.label) for it in self.plan.items
                 if it.label and self.results.get(it.name, {}).get("result")]
        return compare.summarize(pairs) if pairs else None

    def results_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["name", "result_json"])
        for it in self.plan.items:
            r = self.results.get(it.name, {}).get("result")
            w.writerow([it.name, json.dumps(r, ensure_ascii=False) if r else ""])
        return buf.getvalue()

    def results_json(self) -> dict:
        return {
            "job_id": self.id, "created": self.created, "finished": self.finished,
            "predictions": [
                {"name": it.name, "status": it.status, "error": it.error, "warnings": it.warnings,
                 "result": self.results.get(it.name, {}).get("result"),
                 "reasons": self.results.get(it.name, {}).get("reasons"),
                 "audio_seconds": self.results.get(it.name, {}).get("duration"),
                 "processing_seconds": self.results.get(it.name, {}).get("elapsed")}
                for it in self.plan.items
            ],
            "validation": {k: v for k, v in self.plan.to_dict().items() if k != "items"},
            "comparison": self.comparison() if self.plan.has_labels else None,
        }


class JobManager:
    def __init__(self):
        os.makedirs(JOBS_DIR, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self.models_ready = threading.Event()
        self.model_error = None
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()

    def new_upload_dir(self) -> str:
        d = os.path.join(JOBS_DIR, "upload_" + uuid.uuid4().hex[:12])
        os.makedirs(d)
        return d

    def submit(self, plan: BatchPlan, upload_dir: str, source: str) -> Job:
        job = Job(plan, upload_dir, source)
        self.jobs[job.id] = job
        self.q.put(job.id)
        return job

    def queue_position(self, job_id: str) -> int:
        return sum(1 for j in self.jobs.values() if j.state == "queued" and j.created < self.jobs[job_id].created)

    # ---- worker ------------------------------------------------------------
    def warmup(self):
        try:
            from pipeline.analyzer import warmup

            warmup()
        except Exception as e:
            self.model_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.models_ready.set()

    def _worker(self):
        self.warmup()
        while True:
            job = self.jobs.get(self.q.get())
            if job is None:
                continue
            try:
                self._run(job)
            except Exception as e:
                job.state, job.error = "failed", f"internal error: {e}"
                traceback.print_exc()
            finally:
                job.finished = time.time()
                shutil.rmtree(job.upload_dir, ignore_errors=True)

    def _run(self, job: Job):
        from pipeline.analyzer import analyze

        job.state, job.started = "running", time.time()
        for it in job.plan.items:
            if it.status != "pending":
                continue
            with job.lock:
                it.status, job.current = "running", it.name
            t0 = time.time()
            try:
                out = analyze(it.path)
                rec = {"result": out["result"], "reasons": out["reasons"],
                       "duration": out["features"]["duration"], "elapsed": round(time.time() - t0, 1)}
                status, err = "done", None
            except AudioError as e:
                rec, status, err = {"elapsed": round(time.time() - t0, 1)}, "failed", str(e)
            except MemoryError:
                rec, status, err = {"elapsed": round(time.time() - t0, 1)}, "failed", "file too large to analyse (out of memory)"
            except Exception as e:
                traceback.print_exc()
                rec, status, err = {"elapsed": round(time.time() - t0, 1)}, "failed", f"analysis error: {type(e).__name__}: {e}"
            with job.lock:
                job.results[it.name] = rec
                it.status, it.error = status, err
                job.audio_seconds += rec.get("duration") or 0.0
            try:
                os.remove(it.path)  # audio is deleted as soon as it has been analysed
            except OSError:
                pass
        job.current = None
        job.state = "done"

    def _janitor(self):
        while True:
            time.sleep(600)
            now = time.time()
            for jid, job in list(self.jobs.items()):
                if job.finished and now - job.finished > JOB_TTL_SECONDS:
                    self.jobs.pop(jid, None)
