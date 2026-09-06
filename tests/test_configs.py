"""Every config in this repo is well-formed and points at things that exist.

What this file deliberately does not do is validate against alidade's schema. A
hand-written restatement would drift the moment the real one moved, and running
the real parser needs alidade installed.

Giving this repo a token to install it was considered and rejected. Fork pull
requests get no secrets, but a branch pull request does, so the token would be
exactly as protected as write access to this public repo. Granting a contributor
write access here would silently grant them read access to the engine.

Schema validation therefore lives in alidade, whose contract suite fetches these
files. The gap that leaves is real and worth naming: adding a config here is not
checked against the engine on the pull request that adds it.

So the division is: alidade decides whether a config is legal, and this repo
decides whether it is honest about its own contents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
THIS_REPO = "alidade-ml/alidade-templates"

# The one directory holding alidade configs: files you hand to `alidade submit`.
# Named rather than globbed, because the other YAML here is model and trainer
# hyperparameters that alidade never parses.
#
# One directory rather than two. The canary config used to sit beside the script
# it runs, which the training template does not do, so the repo taught two
# different habits. Everything alidade parses now lives in one place and
# everything else is code.
CONFIG_DIRS = ("experiments",)
CONFIGS = sorted(c for d in CONFIG_DIRS for c in (REPO / d).glob("*.yaml"))


def _ids(paths):
    return [str(p.relative_to(REPO)) for p in paths]


def test_this_repo_actually_ships_configs():
    """Guards the glob. An empty CONFIGS list would make every test below pass
    by iterating nothing."""
    assert CONFIGS, "no configs found; the parametrised tests would be vacuous"


@pytest.mark.parametrize("config", CONFIGS, ids=_ids(CONFIGS))
def test_it_parses(config: Path):
    loaded = yaml.safe_load(config.read_text())
    assert isinstance(loaded, dict), f"{config.name} is not a YAML mapping"
    assert "experiment" in loaded, f"{config.name} declares no experiment"


@pytest.mark.parametrize("config", CONFIGS, ids=_ids(CONFIGS))
def test_every_path_it_runs_is_in_this_repo(config: Path):
    """A config whose steps name a file that is not here fails on the compute
    host, after an instance has been paid for."""
    import re

    loaded = yaml.safe_load(config.read_text())
    exp = loaded.get("experiment", {})
    if exp.get("repo", "") and THIS_REPO not in exp["repo"]:
        pytest.skip(f"{config.name} runs against another repo, so its paths are not ours")

    commands = "\n".join(s.get("command", "") for s in exp.get("steps", []))
    referenced = sorted(
        set(re.findall(r"(?<![\w/.-])([\w-]+(?:/[\w.-]+)+\.(?:py|txt))", commands))
    )
    assert referenced, f"{config.name} names no runnable file in its steps"

    missing = [p for p in referenced if not (REPO / p).exists()]
    assert not missing, f"{config.name} runs {missing}, which are not in this repo"


def test_the_canary_config_points_at_this_repo():
    """alidade fetches this file and submits it. If its repo drifts to somewhere
    else, the canary silently stops testing the template it came from."""
    exp = yaml.safe_load((REPO / "experiments" / "canary.yaml").read_text())["experiment"]
    assert THIS_REPO in exp["repo"], exp["repo"]
    assert exp.get("push_tags") is False, (
        "the canary must not push tags: nobody running it can write to this repo"
    )


def test_trainer_config_is_not_an_alidade_config():
    """The two schemas are separate on purpose.

    training/configs/ holds model and trainer hyperparameters, read by
    train_bert.py. alidade never parses it. Conflating the two is the beginner
    failure this repo's layout exists to prevent, so it is pinned rather than
    left to convention.
    """
    trainer_configs = sorted((REPO / "training" / "configs").glob("*.yaml"))
    assert trainer_configs, "no trainer configs found; this check would be vacuous"

    for path in trainer_configs:
        loaded = yaml.safe_load(path.read_text())
        assert "experiment" not in loaded, (
            f"{path.name} declares an alidade `experiment:` block. Either it is "
            f"in the wrong directory, or an alidade config leaked into the "
            f"trainer's hyperparameters."
        )
        assert {"model", "training"} <= set(loaded), sorted(loaded)
