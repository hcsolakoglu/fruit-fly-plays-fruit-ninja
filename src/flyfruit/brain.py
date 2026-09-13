"""Fixed, signed MaleCNS reservoir. Readout receives descending activity only."""

from pathlib import Path
import numpy as np
from scipy import sparse
from .vision import encode, motor

ROOT = Path(__file__).resolve().parents[2]


class Brain:
    def __init__(self, condition="intact", root=ROOT):
        d = Path(root) / "data"
        self.condition = condition
        self.w = sparse.load_npz(d / "connectome.npz")
        interface = np.load(d / "interface.npz")
        self.inputs = interface["input_idx"]
        self.outputs = interface["output_idx"]
        self.bins = interface["sensor_bins"]
        self.iw = interface["sensor_weights"]
        if condition == "rewired":
            self.w = sparse.load_npz(d / "rewired.npz")
        if condition == "disconnected":
            self.w = self.w * 0
        name = "readout_rewired.npz" if condition == "rewired" else "readout.npz"
        r = np.load(d / name)
        self.coef = r["coef"]
        self.mean = r["mean"]
        self.std = r["std"]
        self.state = np.zeros(self.w.shape[0], np.float32)
        self.rng = np.random.default_rng(987)

    def propagate(self, features):
        drive = np.zeros(self.w.shape[0], np.float32)
        drive[self.inputs] = (features[self.bins] * self.iw).sum(1) * 3
        h = np.zeros_like(drive)
        for _ in range(6):
            h = np.tanh(1.6 * (self.w @ h) + drive)
        self.state = h
        return h

    def act(self, rgb, blade):
        f = encode(rgb)
        if self.condition == "blind":
            f = f * 0
        h = self.propagate(f)
        if self.condition == "random":
            field = np.zeros_like(f)
            action = (
                float(self.rng.uniform(40, 600)),
                float(self.rng.uniform(40, 440)),
                True,
            )
        elif self.condition == "pixel":
            field = f
            action = motor(field, blade)
        else:
            z = (h[self.outputs] - self.mean) / self.std
            field = z @ self.coef[:-1] + self.coef[-1]
            action = motor(field, blade)
        return action, {"activity": h, "field": field, "retina": f}
