"""A runnable demo experiment: open a run, emit metrics, close.

Deliberately trivial: no model, no dataset, no GPU. It exists so an install can
be verified end to end in seconds. Alidade's canary runs this exact file.

Everything that identifies the run (Aim address, experiment name, tags) comes
from the environment the orchestrator sets. That is what the callback library is
for; nothing here reads an env var by hand.
"""

from __future__ import annotations

import math
import time

from alidade_callbacks import Run

STEPS = 100


def main() -> int:
    with Run() as run:
        for step in range(STEPS):
            loss = 4.0 * math.exp(-step / 30.0) + 0.05 * math.sin(step * 0.3)
            run.log_train(loss=loss, step=step)
            run.log_eval(loss=loss + 0.1, step=step)
            if step % 10 == 0:
                # Without this the whole curve lands inside ~100ms and a reader
                # polling for metrics can sample before the indexer has caught up.
                time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())