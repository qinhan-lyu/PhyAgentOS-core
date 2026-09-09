#!/usr/bin/env python3
"""One-shot robodojo 0.1.0 install helper (self-contained).

Reads an unsigned schema-v3 index template (direct_download_url placeholders),
generates fresh TOS presigned download URLs on the TARGET machine (requires the
inner-bucket read permission + tosutil), writes a temporary schema-v3 index to a
private temp dir, then calls `paos skill install robodojo --version 0.1.0 --index <idx>`.

Self-contained preparation:
  * locates the template next to this script (downloaded together from TOS) OR in the
    repo layout (<script>/../docs/paos-forge-packages.template.yaml);
  * if `--paos-config` is supplied, prepares the isolated instance so the install does
    NOT depend on the previous acceptance tree: creates <config>.parent/config.json and
    <config>.parent/py/sitecustomize.py (which calls PhyAgentOS set_config_path), and sets
    PAOS_CONFIG + PYTHONPATH for the paos child.

Never modifies PAOS integrity checks, never changes the frozen packages, never
tracks the signed index in git.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLACEHOLDERS = {
    "__SKILL_URL__":   "skill-bundles/robodojo/0.1.0/robodojo-0.1.0.tar.gz",
    "__FORGE_URL__":   "node-bundles/forge_runtime/0.1.0/forge-runtime-0.1.0-linux-x86_64.tar.gz",
    "__ENDPOINT_URL__":"node-bundles/robodojo_endpoint/0.1.0/robodojo-endpoint-0.1.0-linux-x86_64.tar.gz",
}
BUCKET = "phyagentos-resource-inner"

# sitecustomize.py keeps get_data_dir() anchored to the instance dir (config.parent).
SITECUSTOMIZE = '''import os
_p = os.environ.get("PAOS_CONFIG")
if _p:
    try:
        from PhyAgentOS.config.loader import set_config_path
        from pathlib import Path
        set_config_path(Path(_p).expanduser().resolve())
    except Exception:
        pass
'''

def resolve_template(explicit: str | None) -> Path:
    """Locate the unsigned template. Same-directory (TOS download) first, repo layout second."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"template not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    candidates = [
        here / "paos-forge-packages.template.yaml",
        here.parent / "docs" / "paos-forge-packages.template.yaml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise SystemExit(
        "template not found; place paos-forge-packages.template.yaml next to this script "
        "or pass --template <path>"
    )

def find_tosutil(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("TOSUTIL")
    if env:
        return env
    found = shutil.which("tosutil")
    if found:
        return found
    raise SystemExit(
        "tosutil not found. Configure it with --tosutil <path>, "
        "TOSUTIL=<path>, or put 'tosutil' on PATH. "
        "It needs read access to the phyagentos-resource-inner bucket."
    )
def find_paos(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("PAOS_BIN")
    if env:
        return env
    found = shutil.which("paos")
    if found:
        return found
    raise SystemExit(
        "paos not found. Configure it with --paos <path>, "
        "PAOS_BIN=<path>, or put 'paos' on PATH."
    )
def prepare_instance(config_path: Path) -> None:
    """Make an isolated instance so install does not depend on a manually created tree."""
    config_path = config_path.expanduser().resolve()
    data_dir = config_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("{}\n", encoding="utf-8")
    py_dir = data_dir / "py"
    py_dir.mkdir(parents=True, exist_ok=True)
    sc = py_dir / "sitecustomize.py"
    if not sc.exists():
        sc.write_text(SITECUSTOMIZE, encoding="utf-8")


def presign(tosutil: str, key: str, vp: str) -> str:
    r = subprocess.run([tosutil, "presign", f"tos://{BUCKET}/{key}", f"-vp={vp}"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("https://"):
            return line
    raise RuntimeError(f"presign failed for {key}: {r.stdout[:200]} {r.stderr[:200]}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=None, help="unsigned schema-v3 template (default: next to script, then repo docs)")
    ap.add_argument("--tosutil", default=None, help="tosutil path (default: TOSUTIL/PATH/home)")
    ap.add_argument("--paos", default=None, help="paos CLI path (default: PAOS_BIN/PATH/home)")
    ap.add_argument("--vp", default="1d", help="presign validity, e.g. 1d/24h/1440min/86400s")
    ap.add_argument("--out", default=None, help="where to write generated index (default: private temp dir)")
    ap.add_argument("--gen-only", action="store_true", help="generate index only, do not install")
    ap.add_argument("--index", default=None, help="use a pre-generated index file instead of generating")
    ap.add_argument("--skill", default="robodojo")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--paos-config", default=os.environ.get("PAOS_CONFIG"))
    args = ap.parse_args()

    template = resolve_template(args.template)
    text = template.read_text(encoding="utf-8")
    tosutil = find_tosutil(args.tosutil)
    paos = find_paos(args.paos)
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.index:
        idx_path = Path(args.index)
        if not idx_path.is_file():
            raise SystemExit(f"index not found: {idx_path}")
    else:
        urls = {ph: presign(tosutil, key, args.vp) for ph, key in PLACEHOLDERS.items()}
        for ph, url in urls.items():
            if ph not in text:
                raise SystemExit(f"placeholder {ph} missing in template")
            text = text.replace(ph, url)
        out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="robodojo-index-")) / "index.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        idx_path = out
        print(f"[robodojo-helper] index written: {idx_path}")
        print(f"[robodojo-helper] generated_at={generated_at} presign_vp={args.vp}")
        print(f"[robodojo-helper] NOTE: download URLs expire after {args.vp}; re-run to refresh.")

    if args.gen_only:
        print(idx_path)
        return 0

    env = dict(os.environ)
    if args.paos_config:
        cfg = Path(args.paos_config)
        prepare_instance(cfg)
        env["PAOS_CONFIG"] = str(cfg.resolve())
        instance_py = str(cfg.resolve().parent / "py")
        existing_pypath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = instance_py + (os.pathsep + existing_pypath if existing_pypath else "")

    cmd = [paos, "skill", "install", args.skill, "--version", args.version, "--index", str(idx_path)]
    print(f"[robodojo-helper] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    return r.returncode

if __name__ == "__main__":
    sys.exit(main())
