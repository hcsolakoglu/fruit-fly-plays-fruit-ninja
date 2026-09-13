"""CPU-rendered 3D fly-at-PC stage. Body motion is presentation, not a fly physics model."""

import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def make_scene():
    faces = []

    def face(points, color, alpha=1, edge=False, tag=""):
        faces.append(
            {
                "p": np.asarray(points).round(5).tolist(),
                "c": list(color),
                "a": alpha,
                "edge": edge,
                "tag": tag,
            }
        )

    def cube(center, size, color):
        c = np.array(center)
        r = np.array(size) / 2
        v = [
            c + r * np.array([x, y, z])
            for x, y, z in [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ]
        ]
        for q in [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 4, 7, 3),
            (1, 2, 6, 5),
            (3, 7, 6, 2),
            (0, 1, 5, 4),
        ]:
            face([v[i] for i in q], color)

    def sphere(center, radii, color, n=20, m=10, edge=False, tag=""):
        c = np.array(center)
        r = np.array(radii)

        def p(i, j):
            a = i / n * math.tau
            b = -math.pi / 2 + j / m * math.pi
            return c + r * np.array(
                [math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b)]
            )

        for j in range(m):
            for i in range(n):
                face(
                    [p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)],
                    color,
                    edge=edge,
                    tag=tag,
                )

    def rod(a, b, r, color):
        a = np.array(a)
        b = np.array(b)
        v = b - a
        v = v / np.linalg.norm(v)
        u = np.cross(v, [0, 1, 0])
        if np.linalg.norm(u) < 0.01:
            u = np.cross(v, [1, 0, 0])
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)
        for i in range(6):

            def q(j):
                return r * (
                    u * math.cos(j * math.tau / 6) + w * math.sin(j * math.tau / 6)
                )

            face([a + q(i), a + q(i + 1), b + q(i + 1), b + q(i)], color)

    # Desk and computer. The screen quad is textured with the actual running game.
    cube([0, -0.12, 0.1], [6.8, 0.24, 4.5], [29, 39, 57])
    cube([0, -0.251, 0.1], [6.7, 0.03, 4.4], [57, 160, 149])
    for x in [-2.85, 2.85]:
        for z in [-1.65, 1.85]:
            cube([x, -0.85, z], [0.13, 1.5, 0.13], [22, 30, 45])
    cube([0, 0.08, -1.15], [1.25, 0.14, 0.7], [46, 58, 76])
    cube([0, 0.58, -1.38], [0.18, 1.0, 0.16], [58, 73, 95])
    cube([0, 1.75, -1.5], [3.72, 2.88, 0.16], [31, 43, 63])
    cube([0, 1.75, -1.398], [3.58, 2.72, 0.045], [83, 174, 166])
    face(
        [
            [-1.68, 3.01, -1.367],
            [1.68, 3.01, -1.367],
            [1.68, 0.49, -1.367],
            [-1.68, 0.49, -1.367],
        ],
        [15, 25, 40],
        tag="screen",
    )
    sphere([0, 0.385, -1.39], [0.036, 0.036, 0.01], [105, 255, 213], n=8, m=4)
    # Keyboard and mouse, all modeled in three dimensions.
    cube([-0.28, 0.065, -0.04], [1.9, 0.12, 0.66], [16, 24, 38])
    for row in range(4):
        for col in range(12):
            cube(
                [-0.99 + col * 0.135, 0.14, -0.27 + row * 0.135],
                [0.107, 0.055, 0.101],
                [53 + row * 4, 80 + col * 2, 98 + row * 8],
            )
    mouse_start = len(faces)
    sphere([1.05, 0.14, 0.52], [0.19, 0.12, 0.29], [73, 92, 113], n=16, m=8)
    cube([1.05, 0.254, 0.44], [0.027, 0.012, 0.13], [90, 255, 218])
    for f in faces[mouse_start:]:
        f["motion"] = "mouse"
    # Fly: segmented abdomen, thorax, head, compound eyes, antennae and six legs.
    sphere([0, 0.53, 1.18], [0.32, 0.25, 0.55], [155, 103, 53], n=24, m=10)
    for z, r in [(1.48, 0.26), (1.30, 0.31), (1.10, 0.31), (0.92, 0.26)]:
        sphere([0, 0.53, z], [r, 0.255, 0.045], [57, 43, 37], n=16, m=8)
    sphere([0, 0.73, 0.74], [0.32, 0.32, 0.34], [87, 83, 76], n=20, m=10)
    sphere([0, 0.89, 0.26], [0.32, 0.29, 0.30], [124, 97, 63], n=20, m=10)
    for side in [-1, 1]:
        sphere(
            [side * 0.263, 0.95, 0.19],
            [0.177, 0.239, 0.246],
            [207, 62, 54],
            n=16,
            m=10,
            edge=True,
        )
        rod([side * 0.14, 1.12, 0.05], [side * 0.21, 1.27, -0.04], 0.014, [58, 47, 39])
        sphere(
            [side * 0.22, 1.28, -0.045], [0.029, 0.045, 0.025], [130, 102, 64], n=8, m=5
        )
        # Fore, middle, and hind legs: two articulated-looking segments each.
        for z, knee, foot in [
            (0.51, [side * 0.55, 0.37, 0.24], [side * 0.67, 0.15, -0.02]),
            (0.80, [side * 0.61, 0.23, 0.88], [side * 0.79, 0.02, 1.10]),
            (1.10, [side * 0.59, 0.24, 1.49], [side * 0.72, 0.02, 1.84]),
        ]:
            moving = side == 1 and z == 0.51
            if moving:
                knee = [0.62, 0.43, 0.39]
                foot = [1.01, 0.265, 0.52]
            first = len(faces)
            rod([side * 0.22, 0.59, z], knee, 0.028, [77, 63, 46])
            rod(knee, foot, 0.021, [65, 54, 41])
            rod(
                foot,
                [foot[0] + side * 0.09, foot[1] - 0.01, foot[2] + 0.07],
                0.012,
                [43, 40, 36],
            )
            if moving:
                for f in faces[first:]:
                    f["motion"] = "foreleg"
        # Thin transparent wing surfaces with vein geometry.
        wing = [
            [side * 0.12, 0.97, 0.75],
            [side * 0.54, 1.055, 0.84],
            [side * 0.94, 1.045, 1.29],
            [side * 0.90, 1.025, 1.72],
            [side * 0.56, 1.01, 1.96],
            [side * 0.21, 0.98, 1.32],
        ]
        face(wing, [130, 204, 225], 0.38, True, tag="wing")
        for tip in [wing[2], wing[3], wing[4]]:
            rod(wing[0], tip, 0.007, [111, 168, 179])
    # Fine bristles make the silhouette recognizably insect-like.
    for i in range(18):
        a = i * 2.399
        x = 0.27 * math.cos(a)
        z = 0.74 + 0.25 * math.sin(a)
        y = 0.75 + 0.23 * math.sqrt(max(0, 1 - (x / 0.32) ** 2))
        rod([x, y, z], [x * 1.2, y + 0.08, z + 0.025], 0.004, [132, 122, 102])
    obj = {
        "faces": faces,
        "screen_note": "Actual game texture; fly model and leg animation are engineered presentation.",
        "default_camera": {"yaw": 0.50, "pitch": 0.37, "distance": 7.4},
    }
    (ROOT / "data/scene3d.json").write_text(json.dumps(obj, separators=(",", ":")))
    return obj


def projection(points, width, height, yaw=0.5, pitch=0.37, distance=7.4):
    target = np.array([0, 0.95, 0.15])
    cam = target + distance * np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ]
    )
    fw = target - cam
    fw /= np.linalg.norm(fw)
    right = np.cross(fw, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fw)
    q = np.asarray(points) - cam
    z = q @ fw
    focal = width * 0.91
    return np.stack(
        [
            width * 0.5 + (q @ right) * focal / z,
            height * 0.52 - (q @ up) * focal / z,
            z,
        ],
        axis=-1,
    )


class Scene3D:
    def __init__(self):
        self.scene = json.loads((ROOT / "data/scene3d.json").read_text())
        self.faces = self.scene["faces"]

    def render(
        self,
        screen,
        width=1440,
        height=920,
        yaw=0.50,
        pitch=0.37,
        distance=7.4,
        blade=(320, 260),
        frame=0,
    ):
        from .render import background

        im = background(width, height).copy()
        d = ImageDraw.Draw(im, "RGBA")
        items = []
        light = np.array([-0.4, 0.85, 0.5])
        light /= np.linalg.norm(light)
        for face in self.faces:
            p = np.array(face["p"])
            offset = np.array(
                [0.14 * (blade[0] / 640 - 0.5), 0, 0.12 * (blade[1] / 480 - 0.5)]
            )
            if face.get("motion") == "mouse":
                p = p + offset
            if face.get("motion") == "foreleg":
                p = p + np.clip((0.59 - p[:, 1]) / 0.325, 0, 1)[:, None] * offset
            v = projection(p, width, height, yaw, pitch, distance)
            n = np.cross(p[1] - p[0], p[2] - p[0])
            norm = np.linalg.norm(n)
            diff = abs(n @ light) / norm if norm > 1e-8 else 0
            shade = 0.42 + 0.58 * diff
            color = tuple(int(c * shade) for c in face["c"])
            items.append((v[:, 2].mean(), v[:, :2], color, face))
        for _, v, color, face in sorted(items, key=lambda x: -x[0]):
            xy = [tuple(pt) for pt in v]
            if face["tag"] == "screen":
                src = np.array(
                    [
                        [0, 0],
                        [screen.width, 0],
                        [screen.width, screen.height],
                        [0, screen.height],
                    ],
                    float,
                )
                A = []
                B = []
                for (x, y), (u, w) in zip(v, src):
                    A.extend(
                        [
                            [x, y, 1, 0, 0, 0, -u * x, -u * y],
                            [0, 0, 0, x, y, 1, -w * x, -w * y],
                        ]
                    )
                    B.extend([u, w])
                coef = np.linalg.solve(A, B)
                warped = screen.transform(
                    (width, height),
                    Image.Transform.PERSPECTIVE,
                    coef,
                    Image.Resampling.BILINEAR,
                )
                mask = Image.new("L", (width, height))
                ImageDraw.Draw(mask).polygon(xy, fill=255)
                im.paste(warped, (0, 0), mask)
                d = ImageDraw.Draw(im, "RGBA")
            else:
                d.polygon(xy, fill=(*color, int(face["a"] * 255)))
                if face["edge"]:
                    d.line(
                        xy + [xy[0]],
                        fill=(38, 65, 81, 120)
                        if face["tag"] == "wing"
                        else (80, 24, 23, 190),
                        width=1,
                    )
        # A visual mouse motion cue driven by the true blade position.
        p = projection(
            [
                [
                    1.05 + (blade[0] / 640 - 0.5) * 0.14,
                    0.27,
                    0.52 + (blade[1] / 480 - 0.5) * 0.12,
                ]
            ],
            width,
            height,
            yaw,
            pitch,
            distance,
        )[0]
        d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(120, 255, 220, 230))
        return im


if __name__ == "__main__":
    make_scene()
