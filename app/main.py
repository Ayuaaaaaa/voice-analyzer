"""Hosted dashboard: login, batch upload (folder or ZIP + CSV manifest), progress, results, downloads.

Run locally:  uvicorn app.main:app --port 7860
Credentials come from VA_USERNAME / VA_PASSWORD; session cookies are signed with VA_SECRET.
"""
import hmac
import os
import posixpath
import secrets
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .batch import BatchError, build_plan, extract_zip
from .jobs import JobManager

STATIC = os.path.join(os.path.dirname(__file__), "static")
USERNAME = os.environ.get("VA_USERNAME", "autoace")
PASSWORD = os.environ.get("VA_PASSWORD", "change-me")
SECRET = os.environ.get("VA_SECRET") or secrets.token_hex(32)
SESSION_COOKIE = "va_session"
SESSION_SECONDS = 12 * 3600
MAX_UPLOAD_BYTES = int(os.environ.get("VA_MAX_UPLOAD_MB", "2048")) * 1024 ** 2
MAX_FILES = 5000

app = FastAPI(title="AutoAce Voice Analyzer", docs_url=None, redoc_url=None)
signer = URLSafeTimedSerializer(SECRET, salt="va-session")
manager = JobManager()
_failed_logins: dict[str, list[float]] = {}


# ---- auth -------------------------------------------------------------------
def _user(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        return signer.loads(token, max_age=SESSION_SECONDS)
    except (BadSignature, SignatureExpired):
        return None


def _require(request: Request):
    if not _user(request):
        raise HTTPException(status_code=401, detail="not logged in")


@app.get("/login")
def login_page(request: Request):
    if _user(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(os.path.join(STATIC, "login.html"))


@app.post("/login")
async def login(request: Request):
    ip = request.client.host if request.client else "?"
    recent = [t for t in _failed_logins.get(ip, []) if time.time() - t < 300]
    if len(recent) >= 10:
        return RedirectResponse("/login?error=locked", status_code=303)
    form = await request.form()
    user, pw = str(form.get("username", "")), str(form.get("password", ""))
    if hmac.compare_digest(user, USERNAME) and hmac.compare_digest(pw, PASSWORD):
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(SESSION_COOKIE, signer.dumps(user), max_age=SESSION_SECONDS, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")
        _failed_logins.pop(ip, None)
        return resp
    _failed_logins[ip] = recent + [time.time()]
    return RedirectResponse("/login?error=1", status_code=303)


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ---- pages ------------------------------------------------------------------
@app.get("/")
def index(request: Request):
    if not _user(request):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "models_ready": manager.models_ready.is_set(), "model_error": manager.model_error}


# ---- jobs -------------------------------------------------------------------
def _safe_rel(name: str) -> str | None:
    rel = posixpath.normpath((name or "").replace("\\", "/").lstrip("/"))
    if not rel or rel == "." or rel.startswith("..") or ":" in rel.split("/")[0]:
        return None
    return rel


@app.post("/api/jobs")
async def create_job(request: Request):
    _require(request)
    form = await request.form(max_files=MAX_FILES, max_fields=MAX_FILES + 10)
    uploads = [f for f in form.getlist("files") if hasattr(f, "filename")]
    if not uploads:
        raise HTTPException(400, "no files were uploaded")

    upload_dir = manager.new_upload_dir()
    saved, total = [], 0
    try:
        for f in uploads:
            rel = _safe_rel(f.filename)
            if rel is None:
                continue
            target = os.path.join(upload_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as out:
                while chunk := await f.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // 1024 ** 2} MB")
                    out.write(chunk)
            saved.append(rel)

        source = "folder"
        if len(saved) == 1 and saved[0].lower().endswith(".zip"):
            source = f"ZIP: {posixpath.basename(saved[0])}"
            zip_path = os.path.join(upload_dir, *saved[0].split("/"))
            extract_dir = os.path.join(upload_dir, "x")
            os.makedirs(extract_dir)
            rels = extract_zip(zip_path, extract_dir)
            os.remove(zip_path)
            plan = build_plan(extract_dir, rels)
        else:
            plan = build_plan(upload_dir, saved)
    except BatchError as e:
        _cleanup(upload_dir)
        raise HTTPException(400, str(e))
    except HTTPException:
        _cleanup(upload_dir)
        raise
    job = manager.submit(plan, upload_dir, source)
    return job.status()


def _cleanup(path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _job(request: Request, job_id: str):
    _require(request)
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (jobs are kept for 24 hours)")
    return job


@app.get("/api/jobs")
def list_jobs(request: Request):
    _require(request)
    jobs = sorted(manager.jobs.values(), key=lambda j: -j.created)
    return [{"id": j.id, "state": j.state, "source": j.source, "created": j.created, "counts": j.counts()} for j in jobs]


@app.get("/api/jobs/{job_id}")
def job_status(request: Request, job_id: str):
    job = _job(request, job_id)
    st = job.status()
    st["queue_position"] = manager.queue_position(job_id) if job.state == "queued" else 0
    st["models_ready"] = manager.models_ready.is_set()
    return st


@app.get("/api/jobs/{job_id}/results.csv")
def download_csv(request: Request, job_id: str):
    job = _job(request, job_id)
    return Response(job.results_csv(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="predictions_{job.id}.csv"'})


@app.get("/api/jobs/{job_id}/results.json")
def download_json(request: Request, job_id: str):
    job = _job(request, job_id)
    return JSONResponse(job.results_json(),
                        headers={"Content-Disposition": f'attachment; filename="predictions_{job.id}.json"'})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if exc.status_code == 401 and not request.url.path.startswith("/api/"):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/robots.txt")
def robots():
    return PlainTextResponse("User-agent: *\nDisallow: /\n")
