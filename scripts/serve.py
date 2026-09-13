"""Local live experiment, with a neural-only controller and explicit baseline modes."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import argparse
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flyfruit.world import World  # noqa: E402 - local source bootstrap
from flyfruit.brain import Brain  # noqa: E402 - local source bootstrap


class Handler(SimpleHTTPRequestHandler):
    world = None
    brain = None
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *args):
        pass

    def json(self, data):
        b = json.dumps(data, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self.path = "/web/index.html"
        if self.path == "/api/config":
            z = np.load(ROOT / "data/display.npz")
            self.json(
                {
                    "coords": z["coords"].tolist(),
                    "pre": z["pre"].tolist(),
                    "rewired_pre": z["rewired_pre"].tolist(),
                    "rewired_post": z["rewired_post"].tolist(),
                    "post": z["post"].tolist(),
                    "nodes": json.loads((ROOT / "data/display_nodes.json").read_text()),
                    "graph": json.loads((ROOT / "data/graph_summary.json").read_text()),
                }
            )
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in ["/api/reset", "/api/step"]:
            self.send_error(404)
            return
        body = json.loads(
            self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1024))
            or b"{}"
        )
        with self.lock:
            if self.path == "/api/reset" or Handler.world is None:
                c = body.get("condition", "intact")
                if c not in [
                    "intact",
                    "rewired",
                    "blind",
                    "disconnected",
                    "random",
                    "pixel",
                ]:
                    self.send_error(400)
                    return
                Handler.world = World(
                    int(body.get("seed", 7)), bool(body.get("hard", False))
                )
                Handler.brain = Brain(c)
            w, b = Handler.world, Handler.brain
            action, trace = b.act(w.pixels(), w.blade)
            events = w.step(action)
            indices = np.load(ROOT / "data/display.npz")["indices"]
            self.json(
                {
                    "state": w.snapshot(),
                    "activity": trace["activity"][indices].tolist(),
                    "retina": trace["retina"].tolist(),
                    "field": trace["field"].tolist(),
                    "events": events,
                    "metrics": w.metrics(),
                    "condition": b.condition,
                }
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8787)
    a = p.parse_args()
    print(f"Open http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
