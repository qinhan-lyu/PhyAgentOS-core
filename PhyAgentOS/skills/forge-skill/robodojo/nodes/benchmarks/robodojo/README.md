# RoboDojo Benchmark Node

This package connects the RoboDojo evaluation environment to Forge through a
two-node Dora dataflow. The benchmark node owns Isaac Sim and the policy node
loads an XPolicyLab policy, exchanging typed observations and named joint
actions. Keep the two nodes in their documented Python environments: Isaac Sim
and XPolicyLab pin incompatible `websockets` versions.

## Portable setup

The repository contains no checkpoint, simulator checkout, proxy setting, or
machine-specific output. All external resources are selected by the deployment
environment (see `profiles/g05/env.template.sh` in the Skill bundle). The g05
profile's `skill.yaml` declares these as `required_environment`:

```text
FORGE_GATEWAY_PYTHON   forge_bench python (contains installable forge-msgs / dora-rs)
FORGE_PKGS_ROOT        dir with forge_gateway/ + forge_tool/ source packages
BENCH_ENDPOINT_PYTHON  forge_bench python
BENCH_ENDPOINT_SRC     dir with bench_endpoint/ package
ROBODOJO_ENV           RoboDojo env (python 3.11, torch 2.7.0+cu128, isaacsim 5.1.0.0, cv2)
ROBODOJO_ROOT          RoboDojo checkout (contains env/ task/ utils/ src/)
XPOLICYLAB_ROOT        XPolicyLab checkout (G05 adapter in policy/G05)
G05_CKPT_PATH          G05 checkpoint file (single ~33.9GB file)
G05_PROCESSOR_PATH     G05 hf_processor directory (config.json/tokenizer.json/...)
CUDA_HOME              CUDA toolkit root (bin/, lib64/)
CUDA_VISIBLE_DEVICES   GPUs visible to the policy process
ROBODOJO_DEVICE_ID     Isaac device index, relative to the visible set
ROBODOJO_CACHE_ROOT    cache root (derives HF_HOME/MODELSCOPE_CACHE/...); optional
ROBODOJO_RESULT_DIR    writable result directory
```

`G05_CKPT_PATH` may point to a released `.pt` file or to its extracted run
directory. `G05_PROCESSOR_PATH` is required for G05 so an upstream deploy file
cannot silently reintroduce a machine-specific processor path; it overrides any
processor path from that file. Select the Isaac device with `device_id` in the
profile config (or `ROBODOJO_DEVICE_ID`). The policy process follows the normal
CUDA visibility rules; there is no hard-coded GPU number or proxy in this node.

Run from this directory after making a private config copy:

```bash
BENCHMARK_CONFIG=/path/to/config.yaml \
ROBODOJO_ROOT=/path/to/robodojo/repo \
dora run dataflow.example.yaml
```

The example uses `put_bottles_into_dustbin`, `arx_x5`, joint actions, and two
layouts as a small smoke run. Increase `init_states_per_task` only when a full
benchmark is explicitly required. Result and video paths are resolved relative
to the node directory and are ignored by git.

## Validation record

The formal Benchmark Node comparison used the public `OpenGalaxea/g05-robodojo`
checkpoint on `put_bottles_into_dustbin` with the same ordered layouts
`layout_id=0..49` on both sides. The public RoboDojo leaderboard reports 94%
success and 96.3% score for this checkpoint/task.

| runner | success | Wilson 95% CI | normalized mean score |
| --- | ---: | --- | --- |
| Official | 46/50 (92%) | [0.8116, 0.9685] | 0.946 |
| Forge | 45/50 (90%) | [0.7864, 0.9565] | 0.934 |

Fisher exact two-sided `p=1.0`; the confidence intervals overlap. Under
Benchmark acceptance manual v3 criterion A this is a **PASS**.

The deterministic bonus probe matched state, metadata, instruction, and image
shape, but image SHA values differed across Isaac processes. Strict whole-record
JSONL equality is therefore **NOT PASS**. The intentional bad-action negative
control produced an immediate state mismatch and is **PASS**.

Do not place raw episode videos, traces, or benchmark result directories in a
commit. Keep only a small summary if an external audit requires one.

## Serve mode (PAOS Skill, g05 profile)

The `profiles/g05/` dataflow runs the same benchmark in **serve mode**: a PAOS
`robodojo` Skill starts `gateway -> bench_endpoint -> robodojo_benchmark -> policy (G05)`
and exposes stable Tools `bench.describe` / `bench.run`. The G05 policy and the
`put_bottles_into_dustbin` task are fixed by the profile; the Tool only selects
`task_ids` / `init_state_ids` / `num_runs` / `seed` / `max_steps`.

The Skill bundle lives at `skill/robodojo/`. Build and install are described in
`skill/robodojo/DELIVERY_NOTES.md`; `scripts/precheck_robodojo_env.py` validates
the deployment environment before starting the Skill.

### max_steps semantics

`max_steps` is a **hard cap on control steps**. The episode is terminated as
`timed_out` if this many steps are reached before the environment ends. The task
native `step_lim` defines its normal end (`success` / `partial` / `failed`). Set
`max_steps` to at least the task horizon (from `bench.describe` defaults) to allow
completion, or lower to force an explicit step-limit stop. The RoboDojo adapter no
longer silently overwrites the requested `max_steps` with the environment
`step_lim`; it logs the check as a diagnostic only. Both `max_steps` and
`num_steps` count control steps (policy actions).

### Video recording

Episode videos are recorded with OpenCV (`cv2`, `mp4v` codec); **ffmpeg is not
required**. If cv2 is unavailable or the codec fails, recording is disabled and a
single log line is emitted; it never fails the batch. Do not rely on `ffmpeg`
being installed or on the root PATH for video output.