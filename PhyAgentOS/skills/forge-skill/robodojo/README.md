# robodojo Skill

`robodojo` is a Forge Skill with one official profile, `g05`. It runs the RoboDojo
`put_bottles_into_dustbin` task through Isaac Sim and exposes two Gateway tools to
PAOS: `bench.describe` and `bench.run`.

## Layout

```text
skill.yaml                     manifest (manifest_version 2, profile g05)
SKILL.md                       Skill document referenced by skill_document
profiles/g05/dataflow.yaml     gateway -> bench_endpoint -> robodojo_benchmark -> policy
profiles/g05/*.yaml            gateway / bench_endpoint / benchmark config
profiles/g05/bin/*             in-bundle launchers for the benchmark and policy nodes
nodes/benchmarks/robodojo/     benchmark node source shipped inside the Skill bundle
scripts/install_robodojo.py    per-run TOS presign + `paos skill install --index`
docs/                          unsigned schema-v3 index template + 0.1.0 delivery record
```

`gateway` and `bench_endpoint` are not shipped in this Skill: they come from the two
independently versioned Node bundles locked in `skill.yaml`:

| node_id | artifact_id | node_digest |
| --- | --- | --- |
| `forge_runtime` | `forge-runtime-0.1.0` | `cee927eb587f3f958c1803086e59692f6f7ae373477b3cfd15118d01d76c408e` |
| `robodojo_endpoint` | `robodojo-endpoint-0.1.0` | `b9a66fb07b04ee1f76bb8221f68f96b35d1f1ce136d3d6cf967cb7a34da8e159` |

## Source / artifact correspondence

- Frozen Skill artifact `robodojo-0.1.0.tar.gz`
  sha256 `5c7806843e5cff16a8a407254ffe7736c9b1d5d13982bd1b82195c7e4f5bc907`
- Frozen runtime source commit `aac600574a6d9dba225216e1f2271e069be2e4e2`
  (branch `qinhan/robodojo-g05-node`, repo `framework/forge_runtime`)
- Install/support material HEAD `1630e75a450464307cdab08de6e7d4536d04af86`

The frozen bundle payload (`SKILL.md`, `skill.yaml`, `profiles/`, `nodes/`) is
byte-identical to this directory; the tracked directory additionally ships
`README.md`, `docs/` and `scripts/`, which the frozen archive does not contain.
So the in-repo run payload matches the published candidate while the install helper
and delivery docs live alongside it. `final_acceptance` remains pending.

## Runtime requirement

`skill.yaml` uses two manifest features that must be present in the PAOS runtime:

- `profiles.<name>.startup_timeout_s` — a per-profile Dora readiness timeout
  (the generic LIBERO `lingbot_va` profile uses `900`).
- digest-locked multi-file Node bundles — `artifacts.nodes.<id>.digest` pins the
  installed `node-manifest.json` digest; the installed manifest's `entrypoints`
  provide `gateway` / `bench_endpoint`.

`profiles.<name>.startup_timeout_s` is already supported by upstream `dev`.
Digest-locked bundles are provided by the `feature/skill-runtime-node-bundles`
runtime change. The legacy single-executable `artifact_type: executable_tar_gz`
lock form keeps working unchanged.

## Build

```bash
python scripts/package_skill.py PhyAgentOS/skills/forge-skill/robodojo \
  --output-dir dist/skills
```

The official packager writes its own `archive-manifest.json`; the rebuilt archive
digest differs from the frozen `5c7806...` because the tracked directory also ships
`README.md`, `docs/` and `scripts/` (the 27 shared payload files are byte-identical).
Use the published TOS object for the frozen `0.1.0` candidate.

## Install

Requires read permission for TOS bucket `phyagentos-resource-inner` and `tosutil`.

```bash
export PAOS_CONFIG=/abs/instance/config.json
python3 scripts/install_robodojo.py --tosutil /path/to/tosutil --vp 1d
```

`install_robodojo.py` regenerates fresh presigned download URLs on every run, writes a
private temporary schema-v3 index, and calls:

```bash
paos skill install robodojo --version 0.1.0 --index <generated-index.yaml>
```

The historical `paos-forge-packages.0.1.0.yaml` carries expired links and is not the
recommended entry point.

## Verify, start, stop

```bash
paos skill list
paos skill inspect robodojo
paos forge-node verify robodojo forge_runtime
paos forge-node verify robodojo robodojo_endpoint

paos skill start robodojo --profile g05
paos skill status robodojo
paos skill stop robodojo
```

`paos skill start` needs `dora` on `PATH`. A running Skill leaves Dora up after the
benchmark finishes; the pending item is that an Agent must actually stop and confirm
when the user asks.

## External environment

The manifest declares `FORGE_GATEWAY_PYTHON`, `BENCH_ENDPOINT_PYTHON`, `ROBODOJO_ENV`,
`ROBODOJO_ROOT`, `XPOLICYLAB_ROOT`, `G05_CKPT_PATH`, `G05_PROCESSOR_PATH`, `CUDA_HOME`,
`CUDA_VISIBLE_DEVICES`, `ROBODOJO_CACHE_ROOT`, `ROBODOJO_DEVICE_ID` and
`ROBODOJO_RESULT_DIR`. `profiles/g05/env.template.sh` lists the same values with
descriptions.

`TASK_NAME=put_bottles_into_dustbin` is also exported by `env.template.sh` and read by
`policy_node.py`; it is not part of the frozen `skill.yaml` `required_environment` list,
so the host must set it explicitly until a new candidate manifest is published.

Checkpoint/processor and Isaac Sim are external and are not bundled here. The G05
checkpoint is ~33.9 GB; do not copy it into the Skill package.
