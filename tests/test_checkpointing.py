"""Offline tests for checkpointing.Run (resume bookkeeping). No dataset, no pretrained weights needed."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CURRENCY_NO_PRETRAINED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tensorflow.keras import layers, models, optimizers

import checkpointing as ckpt


def _tiny_model():
    m = models.Sequential([layers.Input((4,)), layers.Dense(8, activation="relu"), layers.Dense(1, activation="sigmoid")])
    m.compile(optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])
    return m


class _Stop(ckpt.kcb.Callback):
    def __init__(self, after_epoch):
        super().__init__(); self.after = after_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.after:
            raise KeyboardInterrupt("simulated disconnect")


class CheckpointingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.rand(64, 4).astype("float32"); self.y = (self.x.sum(1) > 2).astype("float32")
        self.tmp = tempfile.TemporaryDirectory(); self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, signature="a", fresh=False):
        return ckpt.Run("t", signature={"s": signature}, monitor="val_accuracy", root=self.root, fresh=fresh)

    def _fit_arrays(self, run, epochs, cbs):
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(16)
        return run.fit(_tiny_model(), "train", ds, ds, epochs=epochs, callbacks=cbs, verbose=0)

    def test_interrupted_phase_resumes_and_finishes(self):
        with self.assertRaises(KeyboardInterrupt):
            self._fit_arrays(self._run(), 5, [_Stop(1)])
        run = self._run()                                           # "new process"
        self.assertFalse(run.state["phases"]["train"]["done"])
        # the interruption hit epoch index 1 BEFORE its backup was committed, so only epoch 0 counts as done
        self.assertEqual(run.state["phases"]["train"]["epochs_done"], 1)
        self._fit_arrays(run, 5, [])
        self.assertTrue(run.state["phases"]["train"]["done"])
        self.assertEqual(sorted(run.history("train")["epoch"]), [0, 1, 2, 3, 4])
        self.assertTrue(run.best_path.exists())

    def test_finished_phase_is_skipped(self):
        self._fit_arrays(self._run(), 2, [])
        run = self._run()
        before = run.log_path("train").read_text()
        self._fit_arrays(run, 2, [])
        self.assertEqual(run.log_path("train").read_text(), before)   # nothing was trained again

    def test_changed_dataset_discards_checkpoint(self):
        self._fit_arrays(self._run("a"), 2, [])
        run = self._run("b")                                          # different fingerprint
        self.assertEqual(run.state["phases"], {})
        self.assertFalse(run.best_path.exists())

    def test_fresh_flag_discards_checkpoint(self):
        self._fit_arrays(self._run(), 2, [])
        self.assertEqual(self._run(fresh=True).state["phases"], {})


if __name__ == "__main__":
    unittest.main()
