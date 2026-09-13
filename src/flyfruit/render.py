"""Editorial renderer: every object, blade point and neuron value comes from a trace."""

import json
import math
import functools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
INK = (7, 10, 21)
WHITE = (242, 244, 251)
MUTED = (144, 157, 181)
TEAL = (95, 244, 207)
ORANGE = (255, 153, 94)
PINK = (220, 123, 255)
RED = (255, 88, 115)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


@functools.lru_cache(128)
def font(size, bold=False, mono=False):
    return ImageFont.truetype(
        MONO if mono else BOLD if bold else FONT, max(10, int(size))
    )


def text(im, xy, s, size=28, color=WHITE, bold=False, mono=False):
    ImageDraw.Draw(im).text(xy, str(s), font=font(size, bold, mono), fill=color)


def fit(im, xy, s, size, maxw, color=WHITE, bold=False):
    while font(size, bold).getlength(s) > maxw:
        size -= 1
    text(im, xy, s, size, color, bold)


def line(im, y, w=1080):
    ImageDraw.Draw(im).line((64, y, w - 64, y), fill=(37, 46, 66), width=2)


@functools.lru_cache(8)
def background(w, h):
    yy, xx = np.mgrid[:h, :w]
    r = np.exp(
        -((xx - w * 0.7) ** 2 / (w * 0.9) ** 2 + (yy - h * 0.3) ** 2 / (h * 0.4) ** 2)
    )
    a = np.empty((h, w, 3), np.uint8)
    for c, v in enumerate([6, 10, 21]):
        a[:, :, c] = v + r * [10, 9, 23][c]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for y in range(0, h, 80):
        d.line((0, y, w, y), fill=(15, 21, 36))
    return im


@functools.lru_cache(16)
def sprite(kind, bomb=False):
    size = 180
    yy, xx = np.mgrid[:size, :size]
    x = (xx - 90) / 65
    y = (yy - 95) / 65
    r = x * x + y * y
    inside = r <= 1
    z = np.sqrt(np.maximum(0, 1 - r))
    light = np.clip(0.38 + 0.65 * z - 0.2 * x - 0.25 * y, 0.2, 1.2)
    colors = [(250, 113, 57), (181, 93, 249), (62, 225, 176), (251, 196, 66)]
    c = (109, 120, 147) if bomb else colors[kind]
    arr = np.zeros((size, size, 4), np.uint8)
    for j in range(3):
        arr[:, :, j] = np.clip(light * c[j], 0, 255)
    arr[:, :, 3] = inside * 255
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    if bomb:
        d.arc((84, 4, 131, 60), 175, 320, fill=(239, 180, 90), width=7)
        d.ellipse((116, 8, 128, 20), fill=ORANGE)
        d.ellipse((66, 72, 114, 116), fill=(222, 227, 240))
        d.rectangle((75, 103, 105, 122), fill=(222, 227, 240))
        d.ellipse((74, 85, 86, 100), fill=(40, 43, 62))
        d.ellipse((96, 85, 108, 100), fill=(40, 43, 62))
        d.polygon([(89, 102), (84, 111), (94, 111)], fill=(40, 43, 62))
        d.line((58, 132, 122, 144), fill=(222, 227, 240), width=5)
        d.line((60, 145, 120, 131), fill=(222, 227, 240), width=5)
    else:
        d.line((89, 34, 93, 17), fill=(123, 84, 44), width=7)
        d.ellipse((92, 13, 124, 29), fill=(87, 212, 146))
        d.arc(
            (43, 45, 130, 147),
            190,
            280,
            fill=tuple(min(255, int(v * 1.22)) for v in c),
            width=3,
        )
        d.ellipse((59, 51, 74, 72), fill=tuple(min(255, int(v * 0.5 + 127)) for v in c))
    return im


class Recording:
    def __init__(self, name):
        self.name = name
        d = json.loads((ROOT / "results/recordings" / f"{name}.json").read_text())
        self.meta = d["meta"]
        self.states = d["states"]
        self.events = d["events"]
        self.trace = np.load(ROOT / "results/recordings" / f"{name}.npz")
        self.activity = self.trace["activity"].astype(np.float32)

    def at(self, f):
        return self.states[min(max(int(f), 0), len(self.states) - 1)]


class Painter:
    def __init__(self):
        d = np.load(ROOT / "data/display.npz")
        self.coords = d["coords"]
        self.pre = d["pre"]
        self.rpre = d["rewired_pre"]
        self.rpost = d["rewired_post"]
        self.post = d["post"]
        self.indices = d["indices"]
        self.nodes = json.loads((ROOT / "data/display_nodes.json").read_text())
        self.graph = json.loads((ROOT / "data/graph_summary.json").read_text())
        self.colors = np.array(
            [
                TEAL
                if n["class"] == "visual_projection"
                else ORANGE
                if n["class"] == "descending_neuron"
                else PINK
                for n in self.nodes
            ]
        )

    def game(self, rec, f, width=952, height=714):
        f = int(np.clip(f, 0, len(rec.states) - 1))
        s = rec.at(f)
        scale = width / 640
        im = Image.new("RGB", (width, height), (10, 17, 31))
        d = ImageDraw.Draw(im)
        # Grid and stage mark; display ornaments are excluded from the sensory image.
        d.rectangle((0, 310 * scale, width, height), fill=(10, 14, 25))
        d.line((0, 310 * scale, width, 310 * scale), fill=(47, 66, 88), width=2)
        text(
            im,
            (20, 320 * scale),
            "LAUNCH ZONE  /  BLADE STAYS ABOVE",
            int(11 * scale),
            MUTED,
            mono=True,
        )
        for x in range(0, 641, 80):
            d.line((x * scale, 0, x * scale, height), fill=(20, 30, 47), width=1)
        for y in range(0, 481, 80):
            d.line((0, y * scale, width, y * scale), fill=(20, 30, 47), width=1)
        d.arc(
            (-width * 0.1, height * 0.75, width * 1.1, height * 1.6),
            190,
            350,
            fill=(37, 58, 77),
            width=2,
        )
        for o in s["objects"]:
            if -50 < o["y"] < 530:
                sp = sprite(o["kind"], o["bomb"])
                sz = int(o["r"] * 180 / 65 * scale)
                sp = sp.resize((sz, sz), Image.Resampling.LANCZOS).rotate(
                    o["angle"] * 180 / math.pi, resample=Image.Resampling.BICUBIC
                )
                im.paste(
                    sp,
                    (int(o["x"] * scale - sz / 2), int(o["y"] * scale - sz * 95 / 180)),
                    sp,
                )
        # Deterministic event effects; these never change the experimental trajectory.
        for e in rec.events:
            dt = (s["frame"] - e["frame"]) / 30
            if not 0 <= dt < 0.85:
                continue
            x, y = (
                e["x"] * scale,
                (e["y"] + e["vy"] * dt * 0.22 + 140 * dt * dt) * scale,
            )
            col = RED if e["bomb"] else [ORANGE, PINK, TEAL, (251, 213, 107)][e["kind"]]
            for k in range(12):
                ang = k * 2.399 + e["id"]
                dist = (40 + ((k * 7) % 31)) * dt * scale
                px = x + math.cos(ang) * dist
                py = y + math.sin(ang) * dist
                rad = max(1, int((1 - dt / 0.85) * (3 + k % 3) * scale))
                d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=col)
            if e["bomb"]:
                r = int(dt * 160 * scale)
                d.ellipse(
                    (x - r, y - r, x + r, y + r),
                    outline=tuple(int(c * (1 - dt / 0.85)) for c in RED),
                    width=max(1, int(5 * (1 - dt / 0.85))),
                )
            else:
                sp = sprite(e["kind"])
                sz = int(e["r"] * 180 / 65 * scale)
                sp = sp.resize((sz, sz), Image.Resampling.LANCZOS)
                for side in (-1, 1):
                    half = sp.crop(
                        (0 if side < 0 else sz // 2, 0, sz // 2 if side < 0 else sz, sz)
                    )
                    half = half.rotate(
                        side * dt * 100, expand=True, resample=Image.Resampling.BICUBIC
                    )
                    im.paste(
                        half,
                        (
                            int(x + side * dt * 85 * scale - half.width / 2),
                            int(y - half.height / 2),
                        ),
                        half,
                    )
                text(
                    im, (x - 20, y - 50 * dt * scale), "+1", int(26 * scale), col, True
                )
        points = []
        for k in range(max(0, f - 9), f + 1):
            st = rec.at(k)
            if st["slicing"]:
                points.append(tuple(v * scale for v in st["blade"]))
        if len(points) > 1:
            glow = Image.new("RGB", im.size)
            gd = ImageDraw.Draw(glow)
            gd.line(
                points,
                fill=(43, 103, 109),
                width=max(2, int(14 * scale)),
                joint="curve",
            )
            glow = glow.filter(ImageFilter.GaussianBlur(7))
            from PIL import ImageChops

            im = ImageChops.add(im, glow)
            d = ImageDraw.Draw(im)
            for k in range(1, len(points)):
                d.line(
                    (points[k - 1], points[k]),
                    fill=TEAL,
                    width=max(1, int(k / len(points) * 5 * scale)),
                )
            d.line(points[-2:], fill=WHITE, width=max(1, int(2 * scale)))
        bx, by = np.array(s["blade"]) * scale
        r = 7 * scale
        d.ellipse(
            (bx - r, by - r, bx + r, by + r),
            outline=WHITE if s["slicing"] else MUTED,
            width=2,
        )
        d.rounded_rectangle(
            (20, 18, 175 * scale, 66 * scale), radius=12, fill=(9, 14, 27)
        )
        text(im, (32, 25), f"{s['sliced']:02d}", int(27 * scale), TEAL, True)
        text(im, (84 * scale, 31), "FRUIT", int(12 * scale), MUTED, True)
        if f == len(rec.states) - 1:
            text(
                im,
                (width / 2 - 100 * scale, 30),
                "END OF RUN / HOLD",
                int(12 * scale),
                MUTED,
                True,
            )
        text(
            im,
            (width - 154 * scale, 30),
            f"{s['bombs']:02d}  BOMBS",
            int(16 * scale),
            RED,
            True,
        )
        return im

    def brain(self, rec, f, width=952, height=440):
        f = int(np.clip(f, 0, len(rec.states) - 1))
        im = Image.new("RGB", (width, height), (9, 13, 26))
        d = ImageDraw.Draw(im)
        a = rec.activity[f]
        am = np.clip(np.sqrt(abs(a)) * 3, 0, 1)
        c = self.coords.copy()
        theta = 0.08 * math.sin(f / 180)
        x = c[:, 0] * math.cos(theta) + c[:, 1] * math.sin(theta)
        y = c[:, 2] * 0.66 - c[:, 1] * 0.2
        x = (x + 0.5) * width * 0.82 + width * 0.09
        y = (y + 0.42) * height * 0.82 + height * 0.07
        pre, post = (
            (self.rpre, self.rpost)
            if rec.meta["condition"] == "rewired"
            else (self.pre, self.post)
        )
        for p, q in zip(pre, post):
            v = float(am[p] * 0.8 + am[q] * 0.2)
            co = (int(20 + v * 49), int(28 + v * 76), int(47 + v * 80))
            d.line((x[p], y[p], x[q], y[q]), fill=co, width=1)
        glow = Image.new("RGB", im.size)
        gd = ImageDraw.Draw(glow)
        for i in np.argsort(am)[-100:]:
            v = float(am[i])
            r = 4 + 5 * v
            co = tuple(int(k * v * 0.45) for k in self.colors[i])
            gd.ellipse((x[i] - r, y[i] - r, x[i] + r, y[i] + r), fill=co)
        from PIL import ImageChops

        im = ImageChops.add(im, glow.filter(ImageFilter.GaussianBlur(6)))
        d = ImageDraw.Draw(im)
        for i in np.argsort(am):
            v = float(am[i])
            r = 1.5 + 3 * v
            co = tuple(
                int(v * k + (1 - v) * b) for k, b in zip(self.colors[i], (38, 49, 71))
            )
            d.ellipse((x[i] - r, y[i] - r, x[i] + r, y[i] + r), fill=co)
        text(
            im,
            (24, 18),
            "REWIRED CONTROL ACTIVITY"
            if rec.meta["condition"] == "rewired"
            else "RECORDED NEURAL ACTIVITY",
            18,
            MUTED,
            True,
        )
        text(
            im,
            (24, height - 36),
            f"{len(self.nodes)} shown / {self.graph['nodes']:,} simulated",
            16,
            MUTED,
        )
        text(im, (width - 200, height - 36), "SOMA POSITIONS", 14, MUTED, mono=True)
        return im

    def heat(self, field, width=432, height=208):
        f = np.clip(np.asarray(field).reshape(2, 8, 12), 0, 1)
        im = Image.new("RGB", (width, height), (11, 17, 31))
        d = ImageDraw.Draw(im)
        cw = width / 12
        ch = height / 8
        for y in range(8):
            for x in range(12):
                fruit = min(float(f[0, y, x]) * 3, 1)
                haz = min(float(f[1, y, x]) * 3, 1)
                co = (
                    int(14 + haz * 232 + fruit * 26),
                    int(20 + fruit * 215),
                    int(33 + fruit * 155 + haz * 20),
                )
                co = tuple(min(v, 255) for v in co)
                d.rectangle(
                    (x * cw + 2, y * ch + 2, (x + 1) * cw - 2, (y + 1) * ch - 2),
                    fill=co,
                )
        return im

    def portrait(self, rec, f, style="hero", time=0, total=20, summary=None):
        im = background(1080, 1920).copy()
        d = ImageDraw.Draw(im)
        text(im, (64, 45), "TINY BRAIN / BIG PROBLEM", 21, TEAL, True, True)
        text(
            im,
            (923, 45),
            {"hero": "01", "brain": "02", "hazard": "04", "rewire": "05"}[style],
            23,
            MUTED,
            mono=True,
        )
        headings = {
            "hero": ["I gave a fruit fly", "a tiny sword."],
            "brain": ["Pixels in.", "Neurons out."],
            "hazard": ["Fruit. Bomb.", "Tiny brain."],
            "rewire": ["Does the wiring", "actually matter?"],
        }
        for k, s in enumerate(headings[style]):
            fit(im, (60, 107 + k * 98), s, 78, 960, WHITE, True)
        subtitle = {
            "hero": "A real connectome in a closed game loop.",
            "brain": "No pixel-to-blade shortcut in the neural controller.",
            "hazard": "Recognition is engineered. Mistakes are real.",
            "rewire": "Real connections. A rewired control. Same task.",
        }[style]
        fit(im, (66, 326), subtitle, 26, 952, MUTED)
        line(im, 388)
        if style == "brain":
            im.paste(self.brain(rec, f, 952, 560), (64, 414))
            text(im, (66, 997), "VISUAL INPUT", 19, TEAL, True)
            text(im, (566, 997), "DECODED MOTOR FIELD", 19, ORANGE, True)
            im.paste(self.heat(rec.trace["retina"][f], 444, 200), (64, 1035))
            im.paste(self.heat(rec.trace["field"][f], 444, 200), (572, 1035))
            im.paste(self.game(rec, f, 688, 516), (196, 1265))
        else:
            im.paste(self.game(rec, f), (64, 420))
            if style == "rewire":
                label = (
                    "REWIRED + REFIT"
                    if rec.meta["condition"] == "rewired"
                    else "MEASURED WIRING"
                )
                d.rounded_rectangle((80, 525, 425, 576), radius=10, fill=(13, 20, 35))
                text(
                    im,
                    (100, 539),
                    label,
                    19,
                    PINK if rec.meta["condition"] == "rewired" else TEAL,
                    True,
                )
            if style == "hazard":
                label = "0.55× / ACTUAL REPLAY" if 3 <= time < 14 else "1× / ACTUAL RUN"
                text(im, (100, 540), label, 20, ORANGE, True)
                recent = [
                    e
                    for e in rec.events
                    if e["event"] == "bomb" and 0 <= f - e["frame"] < 24
                ]
                if recent:
                    d.rounded_rectangle(
                        (325, 680, 755, 775), radius=18, fill=(75, 20, 39)
                    )
                    text(im, (369, 701), "BOMB HIT", 47, RED, True)
            text(im, (66, 1160), "PIXELS", 20, TEAL, True)
            text(im, (330, 1160), "→  FIXED CONNECTOME", 20, PINK, True)
            text(im, (783, 1160), "→  BLADE", 20, ORANGE, True)
            im.paste(self.brain(rec, f), (64, 1210))
            if style == "rewire" and time > total - 6 and summary:
                d.rounded_rectangle(
                    (64, 1220, 1016, 1700),
                    radius=24,
                    fill=(13, 20, 35),
                    outline=(54, 67, 87),
                    width=2,
                )
                text(im, (100, 1254), "20 MATCHED UNSEEN SEEDS", 22, TEAL, True)
                for k, (name, label) in enumerate(
                    [
                        ("intact", "Real wiring"),
                        ("rewired", "Rewired + refit"),
                        ("pixel", "Pixel heuristic"),
                    ]
                ):
                    rate = summary[name]["mean_hit_rate"]
                    y = 1325 + k * 102
                    text(im, (100, y), label, 25, WHITE, True)
                    d.rounded_rectangle(
                        (440, y + 5, 440 + 440 * rate, y + 30),
                        radius=8,
                        fill=TEAL if k == 0 else PINK if k == 1 else ORANGE,
                    )
                    text(im, (906, y), f"{rate:.0%}", 25, WHITE, True)
                text(
                    im,
                    (100, 1642),
                    "Fruit hit rate • bomb counts in results",
                    19,
                    MUTED,
                )
            else:
                cards = {
                    "hero": [("4,386", "REAL NEURONS"), ("468,923", "MEASURED EDGES")],
                    "hazard": [
                        (str(rec.at(f)["sliced"]), "FRUIT SLICED"),
                        (str(rec.at(f)["bombs"]), "BOMBS HIT"),
                    ],
                    "rewire": [
                        ("FIXED", "BIOLOGICAL WIRING"),
                        ("FITTED", "ENGINEERED READOUT"),
                    ],
                }
                for k, (num, label) in enumerate(cards[style]):
                    text(im, (68 + k * 502, 1680), num, 45, WHITE, True)
                    text(im, (70 + k * 502, 1743), label, 18, MUTED, True)
        if time < 2.7:
            # Opening hook typography is already large; an actual live trace remains visible.
            text(im, (66, 1830), "A FLY CONNECTOME PLAYS FRUIT NINJA", 20, TEAL, True)
        else:
            captions = {
                "hero": [
                    "Pixels → visual neurons → descending activity → blade.",
                    "Fixed MaleCNS wiring. A fitted motor interface.",
                    "Real simulation. Not a biological fly.",
                ],
                "brain": [
                    "Color channels excite 2,048 visual projection neurons.",
                    "Six propagation steps through measured connections.",
                    "1,314 descending neurons feed the fitted readout.",
                ],
                "hazard": [
                    "A colorful target. A bright gray hazard.",
                    "The blade can stop cutting while it repositions.",
                    "Every hit and every mistake is from the saved run.",
                ],
                "rewire": [
                    "The rewired graph gets its own fitted readout.",
                    "Disconnecting edges tests dependence on the circuit.",
                    "One graph shuffle cannot establish a biological advantage.",
                ],
            }
            cap = captions[style][min(2, int(time / total * 3))]
            fit(im, (66, 1821), cap, 23, 952, TEAL)
        text(
            im,
            (66, 1872),
            "MaleCNS v1.0 • reduced rate model • engineered vision + motor",
            16,
            MUTED,
        )
        d.rectangle((64, 1903, 64 + 952 * min(time / total, 1), 1907), fill=TEAL)
        return im

    def comparison(self, a, b, f, t, total=20):
        im = background(1920, 1080).copy()
        d = ImageDraw.Draw(im)
        text(im, (72, 40), "THE CONNECTION TEST", 22, TEAL, True, True)
        text(im, (65, 90), "Cut the wiring. Watch the blade.", 66, WHITE, True)
        text(
            im,
            (72, 185),
            "Same launches. Same fitted readout. Only synaptic propagation is removed.",
            28,
            MUTED,
        )
        for k, (rec, title, col) in enumerate(
            [(a, "MEASURED WIRING", TEAL), (b, "CONNECTIONS CUT", RED)]
        ):
            x = 72 + k * 928
            text(im, (x, 264), title, 25, col, True)
            im.paste(self.game(rec, f, 856, 642), (x, 310))
            s = rec.at(f)
            text(
                im,
                (x, 975),
                f"{s['sliced']} fruit   /   {s['bombs']} bombs",
                29,
                col,
                True,
            )
        d.line((960, 267, 960, 1018), fill=(45, 58, 78), width=2)
        text(im, (1200, 60), "03 / CAUSAL ABLATION", 20, MUTED, mono=True)
        d.rectangle((72, 1052, 72 + 1776 * t / total, 1056), fill=TEAL)
        return im
