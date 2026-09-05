"""Every config in this repo is well-formed and points at things that exist.

What this file deliberately does not do is validate against alidade's schema.
That needs alidade's parser, alidade is private, and this repo is public, so a
check here would be a hand-written restatement of a schema it cannot see, which
drifts the moment the real one moves. Schema validation lives in alidade, in a
contract-marked test that fetches these files.

So the division is: alidade decides whether a config is legal, and this repo
decides whether it is honest about its own contents.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted(REPO.glob("*/*.yaml"))
THIS_REPO = "alidade-ml/alidade-templates"


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
    exp = yaml.safe_load((REPO / "canary" / "canary.yaml").read_text())["experiment"]
    assert THIS_REPO in exp["repo"], exp["repo"]
    assert exp.get("push_tags") is False, (
        "the canary must not push tags: nobody running it can write to this repo"
    )
