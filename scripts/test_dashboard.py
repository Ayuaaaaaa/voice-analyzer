"""End-to-end test of the dashboard API with a clean batch, a messy batch and rejected uploads.

usage: python scripts/test_dashboard.py <provided_calls_folder> [--url http://127.0.0.1:7860] [--user demo --password demo-pass]
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ffmpeg(src, dst, *args):
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, *args, dst], check=True)


def make_zip(path, files: dict):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, data in files.items():
            if isinstance(data, bytes):
                z.writestr(arc, data)
            else:
                z.write(data, arc)


def manifest(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "result_json"])
    w.writerows(rows)
    return buf.getvalue().encode()


def wait(s, url, job_id, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = s.get(f"{url}/api/jobs/{job_id}").json()
        c = st["counts"]
        print(f"  [{time.time() - t0:5.0f}s] {st['state']:8s} done={c['done']} failed={c['failed']} "
              f"skipped={c['skipped']} pending={c['pending']} current={st['current']} eta={st['eta_seconds']}")
        if st["state"] in ("done", "failed"):
            return st
        time.sleep(10)
    raise TimeoutError(job_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--user", default="demo")
    ap.add_argument("--password", default="demo-pass")
    ap.add_argument("--skip-clean", action="store_true")
    a = ap.parse_args()
    url, tmp = a.url.rstrip("/"), tempfile.mkdtemp(prefix="va_test_")
    calls = {n: os.path.join(a.calls, n) for n in sorted(os.listdir(a.calls)) if n.endswith(".ogg")}
    labels = {r["name"]: r["result_json"] for r in csv.DictReader(open(os.path.join(a.calls, "labels.csv"), encoding="utf-8-sig"))}
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print("== auth")
    anon = requests.Session()
    check(anon.get(f"{url}/api/jobs").status_code == 401, "API refuses anonymous requests")
    check(anon.get(f"{url}/", allow_redirects=False).status_code == 303, "dashboard redirects anonymous users to login")
    r = anon.post(f"{url}/login", data={"username": a.user, "password": "wrong"}, allow_redirects=False)
    check("error" in r.headers.get("location", ""), "wrong password rejected")
    s = requests.Session()
    s.post(f"{url}/login", data={"username": a.user, "password": a.password})
    check(s.get(f"{url}/api/jobs").status_code == 200, "login works")

    print("== rejected uploads")
    bad = os.path.join(tmp, "bad.zip")
    open(bad, "wb").write(b"this is not a zip")
    r = s.post(f"{url}/api/jobs", files=[("files", ("bad.zip", open(bad, "rb")))])
    check(r.status_code == 400 and "corrupt" in r.json()["detail"], f"corrupt ZIP -> 400 ({r.json().get('detail')})")
    nocol = os.path.join(tmp, "nocol.zip")
    make_zip(nocol, {"labels.csv": b"file,label\ncall_001.ogg,x\n", "call_001.ogg": calls["call_001.ogg"]})
    r = s.post(f"{url}/api/jobs", files=[("files", ("nocol.zip", open(nocol, "rb")))])
    check(r.status_code == 400 and "name" in r.json()["detail"], f"CSV without name column -> 400 ({r.json().get('detail')})")

    if not a.skip_clean:
        print("== clean batch: ZIP with a wrapping folder, 3 calls + labels")
        good = os.path.join(tmp, "good.zip")
        make_zip(good, {**{f"evaluation_batch/{n}": p for n, p in calls.items()},
                        "evaluation_batch/labels.csv": open(os.path.join(a.calls, "labels.csv"), "rb").read()})
        r = s.post(f"{url}/api/jobs", files=[("files", ("good.zip", open(good, "rb")))])
        check(r.status_code == 200, "clean ZIP accepted")
        st = wait(s, url, r.json()["id"])
        v = st["validation"]
        check(not v["missing_audio"] and not v["not_in_manifest"] and v["has_labels"], "no validation issues, labels found")
        check(st["counts"]["done"] == 3, "all 3 calls analysed")
        cmp = st["comparison"]
        print("  per-field:", {k: f"{x['correct']}/{x['total']}" for k, x in cmp["per_field"].items()})
        rows = list(csv.DictReader(io.StringIO(s.get(f"{url}/api/jobs/{st['id']}/results.csv").text)))
        check([r["name"] for r in rows] == list(calls), "CSV keeps filenames and order")
        check(all(json.loads(r["result_json"])["emotional_tone"] for r in rows), "CSV result_json is valid JSON")
        js = s.get(f"{url}/api/jobs/{st['id']}/results.json")
        check("attachment" in js.headers.get("content-disposition", ""), "JSON download is an attachment")

    print("== messy batch: folder upload with mixed problems")
    c1 = calls["call_001.ogg"]
    mp3, m4a, mono = (os.path.join(tmp, x) for x in ("clip.mp3", "clip.m4a", "mono.wav"))
    ffmpeg(c1, mp3, "-t", "45", "-ac", "1", "-b:a", "48k")
    ffmpeg(c1, m4a, "-t", "45", "-c:a", "aac", "-b:a", "48k")
    ffmpeg(c1, mono, "-t", "45", "-ac", "1", "-ar", "8000")
    silent, short = os.path.join(tmp, "silent.wav"), os.path.join(tmp, "short.wav")
    sf.write(silent, np.zeros(16000 * 5, dtype="float32"), 16000)
    sf.write(short, (0.1 * np.random.randn(3200)).astype("float32"), 16000)
    folder_files = {
        "batch/labels.csv": manifest([
            ["clip.mp3", ""], ["clip.m4a", ""], ["Mono.WAV", labels["call_001.ogg"]],
            ["silent.wav", ""], ["short.wav", ""], ["corrupt.wav", ""], ["empty.mp3", ""],
            ["notes.txt", ""], ["call_999.ogg", ""], ["clip.mp3", ""], ["badjson.wav", "{not json"],
        ]),
        "batch/clip.mp3": mp3, "batch/clip.m4a": m4a, "batch/mono.wav": mono,
        "batch/silent.wav": silent, "batch/short.wav": short,
        "batch/corrupt.wav": b"RIFF\x00\x00\x00\x00WAVEgarbage" * 10, "batch/empty.mp3": b"",
        "batch/notes.txt": b"hello", "batch/extra_unlisted.wav": mono, "batch/sub/nested.wav": mono,
    }
    multipart = [("files", (arc, open(p, "rb") if isinstance(p, str) else p)) for arc, p in folder_files.items()]
    r = s.post(f"{url}/api/jobs", files=multipart)
    check(r.status_code == 200, f"messy folder accepted ({r.status_code})")
    st = wait(s, url, r.json()["id"])
    v = st["validation"]
    print("  validation:", json.dumps({k: v[k] for k in ("missing_audio", "not_in_manifest", "duplicate_rows", "unsupported", "nested_ignored")}))
    for w in v["warnings"]:
        print("   warning:", w)
    check(v["missing_audio"] == ["call_999.ogg", "badjson.wav"], "missing files reported")
    check(v["not_in_manifest"] == ["extra_unlisted.wav"], "unlisted file reported")
    check(v["duplicate_rows"] == ["clip.mp3"], "duplicate row reported")
    check(v["unsupported"] == ["notes.txt"], "unsupported type reported")
    check(v["nested_ignored"] == ["sub/nested.wav"], "nested file reported")
    by = {f["name"]: f for f in st["files"]}
    for n, f in by.items():
        print(f"   {n:20s} {f['status']:8s} {f['error'] or ''} {f['warnings'] or ''}")
    check(all(by[n]["status"] == "done" for n in ("clip.mp3", "clip.m4a", "Mono.WAV", "extra_unlisted.wav")), "mp3 / m4a / mono 8 kHz wav / case-mismatch analysed")
    check(all(by[n]["status"] == "failed" for n in ("silent.wav", "short.wav", "corrupt.wav", "empty.mp3")), "bad audio fails per file without stopping the batch")
    rows = list(csv.DictReader(io.StringIO(s.get(f"{url}/api/jobs/{st['id']}/results.csv").text)))
    check(len(rows) == len(st["files"]), "CSV has one row per file, failed files with empty result_json")
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
