"""Bonus 3D fly-at-PC film, using an existing real closed-loop recording."""

import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flyfruit.render import Painter, Recording, background, text, TEAL, WHITE, MUTED  # noqa: E402
from flyfruit.scene3d import Scene3D  # noqa: E402
from render_videos import audio  # noqa: E402


def main():
    r = Recording("intact_1")
    p = Painter()
    stage = Scene3D()
    frames = list(range(600))
    sources = ["intact_1"] * 600
    wav = ROOT / "videos/06_fly_at_pc_3d.wav"
    out = ROOT / "videos/06_fly_at_pc_3d.mp4"
    audio(wav, frames, "hero", sources, {"intact_1": r})
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1920x1080",
            "-r",
            "30",
            "-i",
            "-",
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        im = background(1920, 1080).copy()
        screen = p.game(r, f, 640, 480)
        yaw = 0.70 - f / 600 * 0.38
        scene = stage.render(
            screen,
            1480,
            980,
            yaw=yaw,
            pitch=0.36,
            distance=7.8,
            blade=r.at(f)["blade"],
            frame=f,
        )
        im.paste(scene, (440, 70))
        text(im, (70, 60), "TINY BRAIN / DESKTOP EDITION", 20, TEAL, True, True)
        text(im, (64, 155), "A fly.", 75, WHITE, True)
        text(im, (64, 242), "A PC.", 75, WHITE, True)
        text(im, (67, 358), "Actual connectome.", 30, TEAL, True)
        text(im, (67, 405), "Actual game output.", 28, WHITE)
        text(im, (68, 498), "4,386 neurons", 29, WHITE, True)
        text(im, (68, 542), "468,923 measured edges", 22, MUTED)
        im.paste(p.brain(r, f, 420, 230), (48, 620))
        text(im, (68, 892), "The monitor is the real run.", 23, TEAL)
        text(
            im, (68, 945), "3D body + mouse motion: engineered presentation.", 19, MUTED
        )
        text(
            im,
            (68, 998),
            "MaleCNS v1.0 / fixed rate circuit / fitted readout",
            18,
            MUTED,
        )
        if f == 180:
            im.save(ROOT / "videos/06_fly_at_pc_3d.jpg", quality=95)
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    wav.unlink()
    m = {
        "video": str(out.relative_to(ROOT)),
        "source": "intact_1",
        "secondary_source": None,
        "fps": 30,
        "duration": 20,
        "source_frame_per_output_frame": frames,
        "source_recording_per_output_frame": sources,
        "editing": "Continuous real-time source with an orbiting 3D camera. Monitor texture is the rendered actual game trace.",
        "audio": "Original synthesis; Foley from actual source-time event crossings.",
        "neural_visual": "Actual recorded rates in inset. Fly body is a stylized presentation model; foreleg and mouse follow actual blade position, without biological body-simulation claims.",
    }
    out.with_suffix(".sources.json").write_text(json.dumps(m, indent=2))
    print("RENDERED", out.name, flush=True)


if __name__ == "__main__":
    main()
