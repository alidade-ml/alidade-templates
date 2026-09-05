# alidade-templates

[![Demo](https://github.com/alidade-ml/alidade-templates/actions/workflows/demo.yml/badge.svg)](https://github.com/alidade-ml/alidade-templates/actions/workflows/demo.yml)

Runnable starting points for [alidade](https://github.com/alidade-ml/alidade)
experiments. Clone one, change the training loop, submit it.

| Template | What it is |
|---|---|
| [`canary/`](canary/) | The smallest experiment that still exercises every integration point. No model, no dataset, no GPU. Submit it as-is: `alidade submit canary/canary.yaml`. |

Configs live here rather than in the alidade repo, so the engine ships no
experiment definitions and no training code.

## canary

`canary/train.py` opens a run, emits 100 steps of a loss-shaped curve under
`train/` and `val/`, and closes. It finishes in seconds.

It exists so an install can be verified end to end. `alidade admin canary` runs
this exact file on a real GPU instance and then asserts that the metrics, tags,
Slack card, git tag, cost record and dashboard entry all arrived. If you have
just set up a NUC, this is what tells you it works.

It is also the shortest honest example of the integration. Everything that
identifies a run (the Aim address, the experiment name, the tags) comes from the
environment alidade sets, read by `alidade-callbacks`. Nothing here reads an
environment variable by hand, and nothing imports alidade itself. That is the
boundary: training code depends on the callback library, never on the
orchestrator.

### Run it yourself

```bash
pip install -r canary/requirements.txt
ALIDADE_AIM_REPO_PATH=/tmp/aim python canary/train.py
```

Then point Aim at `/tmp/aim` to see the curve.

### Why the pinned wheel

`canary/requirements.txt` names an exact release asset rather than a version
range. `releases/latest` for `alidade-callbacks` still resolves to v1.1.2, which
ships the pre-rename `astrolabe_callbacks` package, so a range would install
something whose import fails.

## What the badge means, and what it does not

`tests/` is scaffolding, not part of the template. It installs the pinned wheel,
runs the demo, and asserts what actually landed in Aim: the run carries the
submit id it was given, both namespaces hold every step, and `wall_time` (the one
metric the callback library synthesizes) is present.

A green badge means **the demo works against the callbacks release this repo
pins.** It does not, and cannot, say anything about a particular version of the
alidade engine, because alidade is a private repo and no public CI can install
it. Engine-side compatibility is covered on the alidade side and by the canary
run itself.

The test environment is built from `alidade_callbacks.contract`, never from
hardcoded strings. That module is a vendored copy of the engine's contract,
checked against it by hash on every callbacks build, which is what lets a result
here carry any weight at all.
