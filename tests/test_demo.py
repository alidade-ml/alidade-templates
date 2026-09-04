"""The demo still works against the callbacks release this repo pins.

Not a unit test of the demo. It runs `canary/train.py` as a subprocess exactly
the way an orchestrator would, then reopens the Aim repo and asserts what
actually landed.

The environment is built from ``alidade_callbacks.contract``, never from string
literals. That vendored module is sha-checked against the engine, so reading the
names from it is what lets a green run here say anything about the engine at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from aim import Repo
from alidade_callbacks import contract

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = REPO_ROOT / "canary" / "train.py"
SUBMIT_ID = "demo-ci-0001"
EXPECTED_STEPS = 100


def _env(aim_path: Path, **overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            contract.ENV_AIM_REPO_PATH: str(aim_path),
            contract.ENV_EXPERIMENT_NAME: "alidade-demo",
            contract.ENV_AIM_RUN_TAGS: f"{contract.TAG_SUBMIT_ID}={SUBMIT_ID}",
            "ALIDADE_CALLBACK_STRICT": "1",
        }
    )
    env.update(overrides)
    return env


def _run_demo(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _aim_cli() -> str:
    """Aim has no `python -m aim`, so the console script is the only route."""
    sibling = Path(sys.executable).parent / "aim"
    found = str(sibling) if sibling.exists() else shutil.which("aim")
    if not found:
        raise RuntimeError(
            "aim console script not found; the reindex below would be skipped "
            "and every assertion in this file would pass against an empty repo"
        )
    return found


def _find_run(aim_path: Path, submit_id: str):
    """Locate a run by the tag the orchestrator supplied.

    Reindexes first. Aim's `iter_runs` walks an index that a writing process
    does not update, so a freshly written run is invisible until it is rebuilt
    so a zero here is otherwise indistinguishable from a repo with no runs, and
    would make every assertion below pass against nothing.
    """
    if not (aim_path / ".aim").exists():
        return None
    subprocess.run(
        [_aim_cli(), "storage", "--repo", str(aim_path), "reindex", "-y"],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    for run in Repo.from_path(str(aim_path)).iter_runs():
        if run.get(contract.TAG_SUBMIT_ID) == submit_id:
            return run
    return None


@pytest.fixture
def aim_repo(tmp_path: Path) -> Path:
    path = tmp_path / "aim"
    path.mkdir()
    return path


class TestTheDemoIsNotSilentlyDoingNothing:
    def test_strict_mode_turns_an_unreachable_aim_into_a_failure(self, aim_repo):
        """The assertion that licenses every other assertion in this file.

        The library degrades to a no-op by default. Without STRICT the demo exits
        0 against a dead Aim, and a suite that only checked the exit code would
        pass while proving nothing.
        """
        env = _env(aim_repo)
        env.pop(contract.ENV_AIM_REPO_PATH)
        # A literal, unlike every other name here: the callbacks library reads
        # "ALIDADE_AIM_URL" from a string in _core.py, not from contract.py.
        env["ALIDADE_AIM_URL"] = "aim://127.0.0.1:1"

        assert _run_demo(env).returncode != 0

    def test_without_strict_the_same_dead_aim_exits_zero(self, aim_repo):
        env = _env(aim_repo, ALIDADE_CALLBACK_STRICT="")
        env.pop(contract.ENV_AIM_REPO_PATH)
        # A literal, unlike every other name here: the callbacks library reads
        # "ALIDADE_AIM_URL" from a string in _core.py, not from contract.py.
        env["ALIDADE_AIM_URL"] = "aim://127.0.0.1:1"

        assert _run_demo(env).returncode == 0
        assert _find_run(aim_repo, SUBMIT_ID) is None


class TestTagsTheOrchestratorSupplies:
    def test_no_run_tags_means_no_submit_id_to_find_it_by(self, aim_repo):
        """What the canary's aim verifier fails on: a run it cannot locate."""
        env = _env(aim_repo)
        env.pop(contract.ENV_AIM_RUN_TAGS)

        assert _run_demo(env).returncode == 0
        assert _find_run(aim_repo, SUBMIT_ID) is None

    def test_the_run_carries_the_submit_id_it_was_given(self, aim_repo):
        assert _run_demo(_env(aim_repo)).returncode == 0

        run = _find_run(aim_repo, SUBMIT_ID)
        assert run is not None, "no run carries the submit id the env supplied"
        assert run.get(contract.TAG_SUBMIT_ID) == SUBMIT_ID


class TestTheMetricsArrive:
    def test_both_namespaces_land_with_every_step(self, aim_repo):
        assert _run_demo(_env(aim_repo)).returncode == 0

        run = _find_run(aim_repo, SUBMIT_ID)
        assert run is not None
        counts = {m.name: len(m.values.sparse_numpy()[1]) for m in run.metrics()}

        for name in ("train/loss", "val/loss"):
            assert name in counts, f"{name} missing; got {sorted(counts)}"
            assert counts[name] == EXPECTED_STEPS, (
                f"{name} holds {counts[name]} values, expected {EXPECTED_STEPS}"
            )

    def test_wall_time_is_synthesized_by_the_library(self, aim_repo):
        """The demo never logs wall_time. Its presence is what distinguishes
        going through the callback from writing to Aim by hand, which is the
        integration this repo exists to keep honest."""
        assert _run_demo(_env(aim_repo)).returncode == 0

        run = _find_run(aim_repo, SUBMIT_ID)
        assert run is not None
        assert "wall_time" in {m.name for m in run.metrics()}
