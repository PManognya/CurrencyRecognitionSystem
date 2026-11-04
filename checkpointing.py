"""Resumable training: if the runtime dies (Colab GPU quota, network drop, laptop sleep) just run the same command again.

What is saved after EVERY epoch (inside config.CKPT_DIR/<run name>/):
    <phase>_backup/        full training state: weights + optimizer state + epoch counter   (keras BackupAndRestore)
    best.weights.h5        best weights so far (by the monitored validation metric)
    <phase>_log.csv        per-epoch metrics, appended across restarts (used for the training curves)
    state.json             which phases are finished, best metric so far, fingerprint of the dataset/config

On restart:
    * finished phases are skipped,
    * an unfinished phase continues from the last completed epoch (same optimizer state, same learning rate),
    * if the dataset / class list / image size changed since the checkpoint was written, the old checkpoint is
      discarded with a warning (it would not match the new data),
    * `--fresh` throws everything away and trains from scratch.

On Google Colab set CURRENCY_CKPT_DIR to a folder on Google Drive, otherwise the checkpoints vanish with the runtime
(see Colab_Training.ipynb).
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from tensorflow.keras import callbacks as kcb

import config


class _StateWriter(kcb.Callback):
    """After each epoch: remember the best metric and how many epochs of this phase are done (atomic write)."""

    def __init__(self, run, phase):
        super().__init__()
        self.run, self.phase = run, phase

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.run.monitor)
        if value is not None and value == value:                      # skip None / NaN
            best = self.run.state["best"]
            better = best is None or (value > best if self.run.mode == "max" else value < best)
            if better:
                self.run.state["best"] = float(value)
        self.run.state["phases"][self.phase]["epochs_done"] = epoch + 1
        self.run.save_state()


class Run:
    """One resumable training run (e.g. 'classifier' or 'detector') made of one or more phases."""

    def __init__(self, name, signature, monitor, mode="max", fresh=False, root=None):
        self.name, self.monitor, self.mode = name, monitor, mode
        self.dir = Path(root or config.CKPT_DIR) / name
        self.state_path = self.dir / "state.json"
        self.best_path = self.dir / "best.weights.h5"
        fingerprint = hashlib.md5(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()

        old = self._read_state()
        if fresh and self.dir.exists():
            print(f"[checkpoint] --fresh: deleting old checkpoint {self.dir}")
            shutil.rmtree(self.dir, ignore_errors=True)
            old = None
        elif old is not None and old.get("fingerprint") != fingerprint:
            print(f"[checkpoint] WARNING: dataset/config changed since the checkpoint in {self.dir} was written -> "
                  "discarding it and training from scratch.")
            shutil.rmtree(self.dir, ignore_errors=True)
            old = None

        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = old or {"fingerprint": fingerprint, "best": None, "phases": {}}
        if old:
            done = [p for p, s in old["phases"].items() if s.get("done")]
            partial = {p: s.get("epochs_done", 0) for p, s in old["phases"].items() if not s.get("done")}
            print(f"[checkpoint] found earlier run in {self.dir}: finished phases={done or 'none'}, "
                  f"unfinished={partial or 'none'}, best {monitor}={old.get('best')}  (use --fresh to start over)")
        else:
            print(f"[checkpoint] new run, checkpoints -> {self.dir}")
        self.save_state()

    # ------------------------------------------------------------------ state file
    def _read_state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return None

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.state_path)                               # atomic: never a half-written state file

    def log_path(self, phase):
        return self.dir / f"{phase}_log.csv"

    def history(self, phase):
        """Per-epoch metrics of a phase as a DataFrame (survives restarts)."""
        df = pd.read_csv(self.log_path(phase))
        return df.drop_duplicates("epoch", keep="last").reset_index(drop=True)

    # ------------------------------------------------------------------ training
    def fit(self, model, phase, train_ds, val_ds, epochs, callbacks=(), **fit_kwargs):
        """model.fit(...) for one phase with checkpointing. Returns the best monitored value so far.
        `model` must already be compiled for this phase. The caller loads `self.best_path` afterwards."""
        ph = self.state["phases"].setdefault(phase, {"done": False, "epochs_done": 0})
        if ph["done"]:
            print(f"[checkpoint] phase '{phase}' already finished - skipping")
            return self.state["best"]
        if ph["epochs_done"]:
            print(f"[checkpoint] phase '{phase}': resuming after epoch {ph['epochs_done']}/{epochs}")

        backup_dir = self.dir / f"{phase}_backup"
        # Order matters: the backup goes LAST - it is the commit point. If the process dies before it, the epoch is simply
        # redone on restart (duplicate log rows are dropped in `history`), so no epoch is ever lost from the records.
        cbs = [
            kcb.CSVLogger(str(self.log_path(phase)), append=True),
            kcb.ModelCheckpoint(str(self.best_path), monitor=self.monitor, mode=self.mode, save_best_only=True,
                                save_weights_only=True, initial_value_threshold=self.state["best"]),
            *callbacks,
            kcb.BackupAndRestore(str(backup_dir), save_freq="epoch", delete_checkpoint=False),
            _StateWriter(self, phase),                  # after the commit, so "epochs_done" is always truthful
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, **fit_kwargs)

        ph["done"] = True
        self.save_state()
        shutil.rmtree(backup_dir, ignore_errors=True)                  # big (weights + optimizer) and no longer needed
        return self.state["best"]

    def export_best(self, model, dest):
        """Load the best weights into `model` and copy them to `dest` (the file the app / webcam loads)."""
        model.load_weights(self.best_path)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.best_path, dest)

