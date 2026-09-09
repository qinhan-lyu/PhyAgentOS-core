# RoboDojo 0.1.0 — Installation (release candidate)

**Status**: candidate. The schema-v3 static-index route is verified (fresh-instance install on
host 120 passed). No HTTP Resource Registry is used; no default Skill version changed. Final
functional acceptance is pending.

## Recommended entry point (credential-enabled, per-run presign)
Use the **unsigned template + refresh helper** — NOT the historical index that carries
pre-signed download URLs. The helper presigns fresh HTTPS download links on the target machine
on every run, so it never depends on a link lifetime. The signed index is written to a private
temp dir and is not tracked in git. This per-run presign route is **established**; a durable
HTTP(S)-hosted index / HTTP Resource Registry is **not a blocker** for this route.

## Verified baseline
- Skill `robodojo-0.1.0.tar.gz` sha256 `5c7806843e5cff16a8a407254ffe7736c9b1d5d13982bd1b82195c7e4f5bc907`
- Node `forge-runtime-0.1.0` node_digest `cee927eb587f3f958c1803086e59692f6f7ae373477b3cfd15118d01d76c408e` (entry `bin/gateway`, 60 files)
- Node `robodojo-endpoint-0.1.0` node_digest `b9a66fb07b04ee1f76bb8221f68f96b35d1f1ce136d3d6cf967cb7a34da8e159` (entry `bin/bench_endpoint`, 27 files)
- Source commit `aac600574a6d9dba225216e1f2271e069be2e4e2` (branch `qinhan/robodojo-g05-node`)
- TOS bucket `phyagentos-resource-inner`; Skill+2 Node bundles live under `skill-bundles/robodojo/0.1.0/` and `node-bundles/{forge_runtime,robodojo_endpoint}/0.1.0/`.

## Prerequisites
- Linux x86_64, CUDA GPU (>=16 GB for G05), Isaac Sim host, EGL/GL (`MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl`), `ffmpeg`, `dora` coordinator/daemon.
- PAOS runtime: `startup_timeout_s` is already supported by upstream `dev`. Digest-locked
  multi-file Node bundle support is provided by `feature/skill-runtime-node-bundles`.
  `paos skill install --index` supports the
  schema-v3 static index. `forge-node install <artifact_id>` needs an HTTP Resource Registry
  URL and is **not** configured — use the `--index` route.
- **TOS read permission for `phyagentos-resource-inner`** plus `tosutil` (this is a
  credential-required install, not a public credential-less one). If the target machine cannot
  obtain this permission, "download-authorization method missing" is a delivery blocker.

## Install helper (regenerates download links on each run)
The template `paos-forge-packages.template.yaml` has placeholder `direct_download_url` values.
`install_robodojo.py` presigns fresh HTTPS links (via `tosutil presign`) and writes a temporary
schema-v3 index, then runs the verified install command. It is **self-contained**: it finds the
template next to itself (or in the repo layout `<script>/../docs/`), and it **prepares the
isolated instance** on the target machine by creating `<config>.parent/config.json` and
`<config>.parent/py/sitecustomize.py` (which anchors `get_data_dir()` to the instance via
`PhyAgentOS.set_config_path`). No manually-created `sitecustomize.py` from a prior acceptance
tree is required. `paos`/`tosutil` are resolved via `--paos`/`--tosutil`, then `PAOS_BIN`/`TOSUTIL`, then `PATH`; no hardcoded host path is assumed, and a missing entry exits with setup guidance.

```bash
# 1) put install_robodojo.py next to paos-forge-packages.template.yaml (e.g. from TOS support/)
# 2) choose an isolated instance config path
export PAOS_CONFIG=/your/instance/config.json

# 3) run the helper (uses tosutil on PATH; override with --tosutil; --paos for paos CLI)
python3 install_robodojo.py --tosutil /path/to/tosutil --vp 1d
```
The helper sets `PAOS_CONFIG` + `PYTHONPATH=<instance>/py` for the `paos` child and creates the
isolation files if absent. Equivalent manual form after links are refreshed:
```bash
paos skill install robodojo --version 0.1.0 --index <generated-index.yaml>
```

## Expired links
Re-run `install_robodojo.py` — it regenerates `direct_download_url` on every run and prints
`generated_at` + the presign validity. You never edit the YAML by hand. The signed index is
written to a private temp dir and is not tracked. Do **not** use the historical
`paos-forge-packages.0.1.0.yaml` (its links are time-limited).

## Verify the install

> In-repo copy note: the frozen TOS support copy of this file lists the node verification
> arguments in the older order; the CLI signature is `paos forge-node verify <skill> <node_id>`,
> as shown below.
```bash
paos skill list
paos skill inspect robodojo
paos forge-node verify robodojo forge_runtime
paos forge-node verify robodojo robodojo_endpoint
```
Expected: `robodojo 0.1.0 / g05 / stopped`, Gateway `http://127.0.0.1:19011`,
tools `bench.describe, bench.run`, both nodes verified.

## External assets (not bundled)
G05 checkpoint/processor + Isaac Sim are external (RoboDojo-sim-arx_x5-joint-0, action_source=fm,
action_steps=32, continuous_action=True; model source OpenGalaxea/g05-robodojo analog). Configure
`G05_CKPT_PATH`, `G05_PROCESSOR_PATH`, `ROBODOJO_ENV`, `ROBODOJO_ROOT`, `XPOLICYLAB_ROOT`,
`CUDA_HOME`, `CUDA_VISIBLE_DEVICES`, `ROBODOJO_DEVICE_ID`, `ROBODOJO_CACHE_ROOT`,
`ROBODOJO_RESULT_DIR`.

## Start a profile (later; not part of this candidate install check)
```bash
paos skill start robodojo --profile g05
paos skill status robodojo
paos skill stop robodojo
```

## Known / pending
- Agent auto-stop after a run is not implemented; benchmark finishing leaves the Skill running.
  This is not treated as an error — the pending item is that the Agent actually stops and confirms
  when the user explicitly asks.
- No HTTP Resource Registry service is configured. The candidate static-index route is verified;
  durable HTTPS hosting of the index + maintainer registration would be for public/long-lived
  publishing and are **not required** for the per-run presign install route above.
- `final_acceptance` is pending; no default-version change, no public index mutation.
