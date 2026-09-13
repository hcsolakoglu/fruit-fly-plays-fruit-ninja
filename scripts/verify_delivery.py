"""Verify media, lineage, dataset counts and the saved closed-loop trace."""

import json
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flyfruit.world import World  # noqa: E402 - local source bootstrap
from flyfruit.brain import Brain  # noqa: E402 - local source bootstrap


def main():
    records = {
        p.stem: json.loads(p.read_text())
        for p in (ROOT / "results/recordings").glob("*.json")
    }
    videos = sorted((ROOT / "videos").glob("*.mp4"))
    assert len(videos) == 5
    checks = []
    for p in videos:
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(p),
                ]
            )
        )
        v = next(x for x in probe["streams"] if x["codec_type"] == "video")
        a = next(x for x in probe["streams"] if x["codec_type"] == "audio")
        m = json.loads(p.with_suffix(".sources.json").read_text())
        frames = m["source_frame_per_output_frame"]
        sources = m["source_recording_per_output_frame"]
        assert int(v["nb_frames"]) == m["duration"] * 30 == len(frames) == len(sources)
        assert (
            v["pix_fmt"] == "yuv420p"
            and v["codec_name"] == "h264"
            and a["codec_name"] == "aac"
        )
        assert (v["width"], v["height"]) in [(1080, 1920), (1920, 1080)]
        assert all(0 <= f < len(records[s]["states"]) for f, s in zip(frames, sources))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"], check=True
        )
        checks.append(
            {
                "file": str(p.relative_to(ROOT)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "width": v["width"],
                "height": v["height"],
                "frames": len(frames),
                "duration": float(probe["format"]["duration"]),
                "bytes": p.stat().st_size,
                "video_codec": v["codec_name"],
                "audio_codec": a["codec_name"],
                "full_decode": "passed",
                "frame_lineage": "passed",
            }
        )
    trace = np.load(ROOT / "results/recordings/intact_1.npz")
    vis = np.load(ROOT / "data/display.npz")["indices"]
    w = World(1)
    b = Brain()
    for i, expected in enumerate(records["intact_1"]["states"]):
        action, t = b.act(w.pixels(), w.blade)
        w.step(action)
        actual = w.snapshot()
        actual["action"] = list(action)
        assert actual == expected, ("snapshot", i)
        np.testing.assert_array_equal(
            t["activity"][vis].astype(np.float16), trace["activity"][i]
        )
        np.testing.assert_array_equal(t["field"].astype(np.float16), trace["field"][i])
    rows = json.loads((ROOT / "results/evaluation.json").read_text())
    assert len(rows) == 130
    for condition in ["intact", "rewired", "disconnected", "blind", "random", "pixel"]:
        assert {
            r["seed"] for r in rows if r["condition"] == condition and not r["hard"]
        } == set(range(300, 320))
    text = (ROOT / "docs/POSTS.md").read_text()
    assert sum(text.count(f"## {i}.") for i in range(1, 6)) == 5
    (ROOT / "results/video_verification.json").write_text(json.dumps(checks, indent=2))
    report = {
        "media_files": 5,
        "full_decodes": "passed",
        "frame_lineage": "passed",
        "recomputed_trace_frames": 600,
        "exact_activity_trace": "passed",
        "held_out_episodes": len(rows),
        "post_concepts": 5,
    }
    (ROOT / "results/delivery_verification.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
