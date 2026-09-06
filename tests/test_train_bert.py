"""The training template actually trains.

Parsing a config proves nothing about whether a user who clones this can get a
run. These train the real model for real steps on a CPU, which is the only
reason the model is small: a template CI cannot run is a template that rots,
and this repo already deleted one config for exactly that.

Fast by construction. The whole file is a few seconds.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.training

# Skip the module rather than fail collecting it. The marker alone is not
# enough: -m filters after collection, and collection imports this file, so a
# job that does not install torch would error before the filter ran.
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from torch_models import TinyBertConfig, TinyBertForMaskedLM  # noqa: E402
from training.train_bert import synthetic_batches, train  # noqa: E402


@pytest.fixture
def cfg():
    return TinyBertConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_layers=2, num_heads=4, max_positions=64)


class TestTheModelKnowsNothingBeforeItLearns:
    def test_initial_loss_is_the_uniform_prior(self, cfg):
        """ln(vocab_size), give or take.

        A materially lower number means the model is reading the answer rather
        than predicting it. That happened here: without weight init, the tied
        head made the logit for the input token dominate and the loss started
        at zero.
        """
        torch.manual_seed(0)
        model = TinyBertForMaskedLM(cfg)
        ids, attention_mask, labels = next(
            synthetic_batches(cfg, steps=1, batch_size=4, seq_len=32)
        )
        _, loss = model(ids, attention_mask, labels)

        assert loss.item() == pytest.approx(math.log(cfg.vocab_size), abs=0.6), (
            f"loss {loss.item():.3f} at init, expected about "
            f"{math.log(cfg.vocab_size):.3f}"
        )

    def test_masked_positions_are_hidden_from_the_model(self, cfg):
        """Every position carrying a label must be a mask token in the input.

        Leak one and the task becomes copying, which trains to zero loss and
        teaches nothing.
        """
        ids, _, labels = next(synthetic_batches(cfg, steps=1, batch_size=4, seq_len=32))
        supervised = labels != -100
        assert supervised.any()
        assert (ids[supervised] == cfg.mask_token_id).all()

    def test_unsupervised_positions_keep_their_token(self, cfg):
        ids, _, labels = next(synthetic_batches(cfg, steps=1, batch_size=4, seq_len=32))
        assert (ids[labels == -100] != cfg.mask_token_id).all()


class TestItLearns:
    def test_the_loss_falls_below_the_uniform_prior(self, tmp_path, monkeypatch):
        """The template's chart has to go down.

        A user's first read of a run is the loss curve. Flat is
        indistinguishable from broken, which is why the synthetic task is a
        repeating motif rather than uniform noise.
        """
        settings = yaml.safe_load(
            (REPO / "training" / "configs" / "bert_tiny.yaml").read_text()
        )
        settings["training"]["steps"] = 150
        config = tmp_path / "fast.yaml"
        config.write_text(yaml.safe_dump(settings))

        monkeypatch.setenv("ALIDADE_AIM_REPO_PATH", str(tmp_path / "aim"))
        monkeypatch.setenv("ALIDADE_CALLBACK_STRICT", "1")
        (tmp_path / "aim").mkdir()

        final = train(config)
        prior = math.log(settings["model"]["vocab_size"])
        assert final < prior - 1.0, (
            f"loss ended at {final:.3f} against a uniform prior of {prior:.3f}; "
            f"the model is not learning the task"
        )


def test_the_script_runs_as_the_experiment_invokes_it(tmp_path):
    """The experiment's step command, not an import.

    An importable module that fails under `python training/train_bert.py` would
    pass every test above and fail on the instance.
    """
    settings = yaml.safe_load(
        (REPO / "training" / "configs" / "bert_tiny.yaml").read_text()
    )
    settings["training"]["steps"] = 20
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(settings))
    aim = tmp_path / "aim"
    aim.mkdir()

    result = subprocess.run(
        [sys.executable, "training/train_bert.py", "--config", str(config)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO),
            "ALIDADE_AIM_REPO_PATH": str(aim),
            "ALIDADE_CALLBACK_STRICT": "1",
            "HOME": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "final loss" in result.stdout
