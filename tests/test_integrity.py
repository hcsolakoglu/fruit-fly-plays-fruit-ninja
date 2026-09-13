import sys
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flyfruit.world import World  # noqa: E402 - local source bootstrap
from flyfruit.vision import encode, motor  # noqa: E402 - local source bootstrap
from flyfruit.brain import Brain  # noqa: E402 - local source bootstrap


class Integrity(unittest.TestCase):
    def test_source_edges_and_normalization(self):
        e = np.load(ROOT / "data/edges.npz")
        w = sparse.load_npz(ROOT / "data/connectome.npz")
        raw = sparse.csr_matrix((e["count"], (e["post"], e["pre"])), shape=w.shape)
        self.assertEqual(raw.nnz, w.nnz)
        expected = sparse.diags(1 / np.maximum(np.asarray(raw.sum(1)).ravel(), 1)) @ raw
        self.assertLess(abs(abs(w) - expected).max(), 1e-6)

    def test_no_input_output_overlap(self):
        i = np.load(ROOT / "data/interface.npz")
        self.assertEqual(len(np.intersect1d(i["input_idx"], i["output_idx"])), 0)

    def test_cut_wiring_removes_all_descending_signal(self):
        b = Brain("disconnected")
        f = np.random.default_rng(14).random(192, dtype=np.float32)
        h = b.propagate(f)
        self.assertGreater(abs(h[b.inputs]).max(), 0)
        self.assertEqual(abs(h[b.outputs]).max(), 0)

    def test_visible_pixel_change_changes_neural_output(self):
        b = Brain()
        f = np.zeros(192, np.float32)
        h = b.propagate(f)[b.outputs].copy()
        f[26] = 0.8
        h2 = b.propagate(f)[b.outputs]
        self.assertGreater(float(np.linalg.norm(h2 - h)), 1e-5)

    def test_identical_seed_different_actions_preserve_launch_schedule(self):
        a = World(333)
        b = World(333)
        for _ in range(150):
            a.step((100, 100, False))
            b.step((500, 400, True))
        self.assertEqual(
            (a.spawned, a.bombs_spawned, a.next_id),
            (b.spawned, b.bombs_spawned, b.next_id),
        )

    def test_deterministic_closed_loop(self):
        def roll():
            w = World(88)
            b = Brain()
            for _ in range(60):
                action, _ = b.act(w.pixels(), w.blade)
                w.step(action)
            return w.snapshot()

        self.assertEqual(roll(), roll())

    def test_blank_pixels_no_cut(self):
        self.assertFalse(motor(encode(World().pixels()))[2])

    def test_rewire_not_same_matrix(self):
        a = sparse.load_npz(ROOT / "data/connectome.npz")
        b = sparse.load_npz(ROOT / "data/rewired.npz")
        self.assertEqual(a.shape, b.shape)
        self.assertGreater((a - b).nnz, a.nnz)


if __name__ == "__main__":
    unittest.main()
