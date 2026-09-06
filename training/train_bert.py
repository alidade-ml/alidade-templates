"""Train the masked-language model in torch_models/.

This is the file an experiment's step command runs, and the file you edit. It
does three things: build a model from a config, train it, and let alidade see
the metrics.

The data here is random tokens, which trains nothing useful and keeps the
template runnable anywhere. Replace `synthetic_batches` with your own loader and
the rest of the file stays as it is.

Metrics reach alidade through `alidade_callbacks.Run`. Everything identifying
the run comes from the environment alidade sets; nothing here reads an
environment variable by hand.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from alidade_callbacks import Run

from torch_models import TinyBertConfig, TinyBertForMaskedLM


def synthetic_batches(cfg: TinyBertConfig, *, steps: int, batch_size: int, seq_len: int):
    """Batches of a repeating motif, masked at 15%.

    Deliberately learnable. Each sequence tiles a short random motif, so a
    hidden token is recoverable from the repetitions around it and the loss
    visibly falls. Uniform random tokens would be more honest as noise and
    useless as a template: the loss would sit flat at ln(vocab_size) forever
    and you could not tell a working pipeline from a broken one.

    Replace this with your own loader. The contract the rest of the file
    depends on is the tuple it yields: input with some positions hidden, an
    attention mask, and labels that are -100 everywhere except those positions.
    """
    special = max(cfg.mask_token_id + 1, 4)
    motif_len = 8
    for _ in range(steps):
        motif = torch.randint(special, cfg.vocab_size, (batch_size, motif_len))
        ids = motif.repeat(1, seq_len // motif_len + 1)[:, :seq_len]

        chosen = torch.rand(ids.shape) < 0.15
        # Guarantee at least one target per batch, or the loss is undefined.
        chosen[:, 0] = True
        labels = torch.where(chosen, ids, torch.full_like(ids, -100))
        masked = torch.where(chosen, torch.full_like(ids, cfg.mask_token_id), ids)
        yield masked, torch.ones_like(ids), labels


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def train(config_path: Path) -> float:
    settings = load_config(config_path)
    model_cfg = TinyBertConfig(**settings["model"])
    training = settings["training"]

    torch.manual_seed(training.get("seed", 0))
    model = TinyBertForMaskedLM(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])

    print(f"training {model.num_parameters():,} parameters on {training['steps']} steps")

    last = math.nan
    with Run() as run:
        for step, (ids, attention_mask, labels) in enumerate(
            synthetic_batches(
                model_cfg,
                steps=training["steps"],
                batch_size=training["batch_size"],
                seq_len=training["seq_len"],
            )
        ):
            _, loss = model(ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            last = loss.item()
            # Whatever you pass lands in alidade under this name. There is no
            # whitelist and no renaming.
            run.log_train(loss=last, step=step)
            if step % training.get("eval_every", 10) == 0:
                run.log_eval(loss=last, step=step)

    print(f"final loss {last:.4f}")
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "bert_tiny.yaml",
        help="model and training hyperparameters",
    )
    args = parser.parse_args()
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
