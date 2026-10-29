"""Turn an uploaded folder or ZIP into a validated batch plan.

Expected shape: audio files at the batch root plus one CSV manifest with a `name`
column (exact filename incl. extension) and an optional `result_json` column.
Nothing here raises for a single bad file; every problem becomes a reported issue.
"""
import csv
import io
import json
import os
import posixpath
import shutil
import zipfile
from dataclasses import asdict, dataclass, field

from pipeline.audio import SUPPORTED_EXTENSIONS

MAX_ZIP_MEMBERS = 5000
MAX_UNCOMPRESSED_BYTES = 4 * 1024 ** 3
IGNORED_PREFIXES = ("__MACOSX/", ".")
IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


class BatchError(Exception):
    """The upload as a whole cannot be used (e.g. corrupt ZIP, nothing to process)."""


@dataclass
class FileItem:
    name: str
    path: str | None
    status: str = "pending"  # pending | skipped | failed | done
    error: str | None = None
    in_manifest: bool = True
    label: dict | None = None
    warnings: list = field(default_factory=list)


@dataclass
class BatchPlan:
    manifest_name: str | None
    items: list
    missing_audio: list = field(default_factory=list)
    not_in_manifest: list = field(default_factory=list)
    duplicate_rows: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    nested_ignored: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    has_labels: bool = False

    def to_dict(self):
        d = asdict(self)
        for it in d["items"]:
            it.pop("path", None)
        return d


def _ignored(rel: str) -> bool:
    base = posixpath.basename(rel)
    return (base in IGNORED_NAMES or base.startswith("._") or rel.startswith("__MACOSX/")
            or any(part.startswith(".") for part in rel.split("/")))


def _strip_common_root(paths: list[str]) -> dict[str, str]:
    """Map original relative path -> path relative to the batch root.

    A single wrapping folder (e.g. `evaluation_batch/...`) is treated as the root.
    """
    parts = [p.split("/") for p in paths]
    root_depth = 0
    while parts and all(len(p) > root_depth + 1 for p in parts) and len({p[root_depth] for p in parts}) == 1:
        root_depth += 1
    return {p: "/".join(p.split("/")[root_depth:]) for p in paths}


def extract_zip(zip_path: str, dest: str) -> list[str]:
    """Safely extract a ZIP, returning relative POSIX paths of extracted files."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise BatchError("the uploaded ZIP file is corrupt or not a ZIP archive")
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            raise BatchError(f"ZIP has too many files ({len(members)}; limit {MAX_ZIP_MEMBERS})")
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise BatchError("ZIP is too large when uncompressed (limit 4 GB)")
        out = []
        for m in members:
            rel = m.filename.replace("\\", "/").lstrip("/")
            if _ignored(rel):
                continue
            norm = posixpath.normpath(rel)
            if norm.startswith("..") or os.path.isabs(norm):
                continue
            target = os.path.join(dest, *norm.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as e:
                raise BatchError(f"could not extract '{rel}' from ZIP ({e})")
            out.append(norm)
        return out


def _read_manifest(path: str) -> tuple[list[dict], list[str], bool]:
    """Returns (rows, warnings, has_result_json_column)."""
    warnings = []
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise BatchError("the CSV manifest is not valid text (unknown encoding)")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
    if "name" not in cols:
        raise BatchError(f"the CSV manifest has no 'name' column (found: {', '.join(reader.fieldnames or []) or 'none'})")
    has_json = "result_json" in cols
    if not has_json:
        warnings.append("manifest has no 'result_json' column; predictions will not be compared with labels")
    rows = []
    for i, r in enumerate(reader, start=2):
        name = (r.get(cols["name"]) or "").strip()
        if not name:
            if any((v or "").strip() for v in r.values() if isinstance(v, str)):
                warnings.append(f"CSV line {i}: empty 'name' value, row ignored")
            continue
        label = None
        raw_json = (r.get(cols["result_json"]) or "").strip() if has_json else ""
        if raw_json:
            try:
                label = json.loads(raw_json)
                if not isinstance(label, dict):
                    raise ValueError("not a JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                warnings.append(f"CSV line {i} ({name}): result_json is not valid JSON ({e}); no comparison for this file")
                label = None
        rows.append({"name": name, "label": label, "line": i})
    return rows, warnings, has_json


def build_plan(root: str, rel_paths: list[str]) -> BatchPlan:
    """root: folder holding the uploaded files; rel_paths: their POSIX paths relative to root."""
    rel_paths = [p for p in rel_paths if not _ignored(p)]
    if not rel_paths:
        raise BatchError("the upload contains no files")
    mapping = _strip_common_root(rel_paths)
    files_at_root, nested = {}, []
    for orig, rel in mapping.items():
        if "/" in rel:
            nested.append(rel)
        else:
            files_at_root[rel] = os.path.join(root, *orig.split("/"))

    csvs = sorted(n for n in files_at_root if n.lower().endswith(".csv"))
    audio_like = {n: p for n, p in files_at_root.items() if not n.lower().endswith(".csv")}
    plan = BatchPlan(manifest_name=None, items=[], nested_ignored=sorted(nested))
    if nested:
        plan.warnings.append(f"{len(nested)} file(s) inside sub-folders were ignored; audio must be at the batch root")

    rows = []
    if not csvs:
        plan.warnings.append("no CSV manifest found; all audio files at the batch root will be processed")
    else:
        chosen, errors = None, []
        for c in csvs:
            try:
                rows, w, has_json = _read_manifest(files_at_root[c])
                chosen = c
                plan.warnings.extend(w)
                break
            except BatchError as e:
                errors.append(f"{c}: {e}")
        if chosen is None:
            raise BatchError("no usable CSV manifest: " + "; ".join(errors))
        plan.manifest_name = chosen
        if len(csvs) > 1:
            plan.warnings.append(f"several CSV files found; using '{chosen}' as the manifest")

    lower_index = {n.lower(): n for n in audio_like}
    seen, matched = set(), set()
    for r in rows:
        name = r["name"]
        if name in seen:
            plan.duplicate_rows.append(name)
            continue
        seen.add(name)
        actual = name if name in audio_like else lower_index.get(name.lower())
        if actual is None:
            plan.missing_audio.append(name)
            plan.items.append(FileItem(name=name, path=None, status="skipped",
                                       error="listed in the CSV manifest but no such file was uploaded", label=r["label"]))
            continue
        matched.add(actual)
        item = FileItem(name=name, path=audio_like[actual], label=r["label"])
        if actual != name:
            item.warnings.append(f"matched uploaded file '{actual}' ignoring letter case")
        plan.items.append(item)

    for n in sorted(set(audio_like) - matched):
        item = FileItem(name=n, path=audio_like[n], in_manifest=not rows)
        if rows:
            plan.not_in_manifest.append(n)
            item.warnings.append("not listed in the CSV manifest")
        plan.items.append(item)

    for it in plan.items:
        if it.path and os.path.splitext(it.name)[1].lower() not in SUPPORTED_EXTENSIONS:
            plan.unsupported.append(it.name)
            it.status = "failed"
            it.error = f"unsupported file type '{os.path.splitext(it.name)[1] or '(none)'}'"
    plan.has_labels = any(it.label for it in plan.items)
    if not any(it.status == "pending" for it in plan.items):
        if not plan.items:
            raise BatchError("no audio files found in the upload")
    return plan
