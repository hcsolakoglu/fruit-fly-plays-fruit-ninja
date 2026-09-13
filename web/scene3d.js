/* Perspective 3D mesh renderer on Canvas2D: CPU geometry, no WebGL dependency. */
class FlyDeskScene {
  constructor(canvas, texture) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.texture = texture;
    this.yaw = 0.5;
    this.pitch = 0.37;
    this.distance = 7.4;
    this.data = null;
    this.frame = 0;
    this.blade = [320, 260];
    this.drag = null;
    fetch("/data/scene3d.json")
      .then((r) => r.json())
      .then((d) => {
        this.data = d;
        this.draw();
      });
    canvas.addEventListener("pointerdown", (e) => {
      this.drag = [e.clientX, e.clientY, this.yaw, this.pitch];
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointerup", () => (this.drag = null));
    canvas.addEventListener("pointermove", (e) => {
      if (!this.drag) return;
      this.yaw = this.drag[2] + (e.clientX - this.drag[0]) * 0.005;
      this.pitch = Math.max(
        0.08,
        Math.min(0.85, this.drag[3] + (e.clientY - this.drag[1]) * 0.004),
      );
      this.draw();
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.distance = Math.max(
          5.2,
          Math.min(11, this.distance + e.deltaY * 0.005),
        );
        this.draw();
      },
      { passive: false },
    );
  }
  project(p) {
    let y = this.yaw,
      a = this.pitch,
      d = this.distance,
      cam = [
        Math.sin(y) * Math.cos(a) * d,
        0.95 + Math.sin(a) * d,
        0.15 + Math.cos(y) * Math.cos(a) * d,
      ];
    let q = [p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]],
      right = [Math.cos(y), 0, -Math.sin(y)],
      up = [
        -Math.sin(y) * Math.sin(a),
        Math.cos(a),
        -Math.cos(y) * Math.sin(a),
      ],
      fw = [
        -Math.sin(y) * Math.cos(a),
        -Math.sin(a),
        -Math.cos(y) * Math.cos(a),
      ];
    let dot = (v, w) => v.reduce((s, x, i) => s + x * w[i], 0),
      z = dot(q, fw),
      f = this.canvas.width * 0.91;
    return [
      this.canvas.width * 0.5 + (dot(q, right) * f) / z,
      this.canvas.height * 0.52 - (dot(q, up) * f) / z,
      z,
    ];
  }
  triangle(dst, src) {
    let c = this.ctx;
    let [p0, p1, p2] = src,
      [q0, q1, q2] = dst;
    let den =
      p0[0] * (p1[1] - p2[1]) +
      p1[0] * (p2[1] - p0[1]) +
      p2[0] * (p0[1] - p1[1]);
    if (Math.abs(den) < 1e-8) return;
    let coeff = (v) => [
      (v[0] * (p1[1] - p2[1]) +
        v[1] * (p2[1] - p0[1]) +
        v[2] * (p0[1] - p1[1])) /
        den,
      (v[0] * (p2[0] - p1[0]) +
        v[1] * (p0[0] - p2[0]) +
        v[2] * (p1[0] - p0[0])) /
        den,
      (v[0] * (p1[0] * p2[1] - p2[0] * p1[1]) +
        v[1] * (p2[0] * p0[1] - p0[0] * p2[1]) +
        v[2] * (p0[0] * p1[1] - p1[0] * p0[1])) /
        den,
    ];
    let x = coeff([q0[0], q1[0], q2[0]]),
      y = coeff([q0[1], q1[1], q2[1]]);
    c.save();
    c.beginPath();
    const mid = [(q0[0] + q1[0] + q2[0]) / 3, (q0[1] + q1[1] + q2[1]) / 3];
    const clip = [q0, q1, q2].map((p) => {
      const dx = p[0] - mid[0],
        dy = p[1] - mid[1],
        len = Math.hypot(dx, dy);
      return [p[0] + dx / len, p[1] + dy / len];
    });
    c.moveTo(...clip[0]);
    c.lineTo(...clip[1]);
    c.lineTo(...clip[2]);
    c.closePath();
    c.clip();
    c.setTransform(x[0], y[0], x[1], y[1], x[2], y[2]);
    c.drawImage(this.texture, 0, 0);
    c.restore();
  }
  screen(face) {
    let p = face.p;
    for (let j = 0; j < 6; j++)
      for (let i = 0; i < 8; i++) {
        let uv = [
            [i / 8, j / 6],
            [(i + 1) / 8, j / 6],
            [(i + 1) / 8, (j + 1) / 6],
            [i / 8, (j + 1) / 6],
          ],
          dst = uv.map(([u, v]) =>
            this.project(
              p[0].map((x, k) => x + (p[1][k] - x) * u + (p[3][k] - x) * v),
            ).slice(0, 2),
          ),
          src = uv.map(([u, v]) => [
            u * this.texture.width,
            v * this.texture.height,
          ]);
        this.triangle([dst[0], dst[1], dst[2]], [src[0], src[1], src[2]]);
        this.triangle([dst[0], dst[2], dst[3]], [src[0], src[2], src[3]]);
      }
  }
  draw() {
    if (!this.data) return;
    let c = this.ctx,
      w = this.canvas.width,
      h = this.canvas.height,
      g = c.createRadialGradient(
        w * 0.5,
        h * 0.45,
        10,
        w * 0.5,
        h * 0.5,
        w * 0.7,
      );
    g.addColorStop(0, "#182337");
    g.addColorStop(1, "#080d19");
    c.fillStyle = g;
    c.fillRect(0, 0, w, h);
    let faces = this.data.faces
      .map((f) => {
        let points = f.p.map((p) => {
          let q = p.slice(),
            weight =
              f.motion === "mouse"
                ? 1
                : f.motion === "foreleg"
                  ? Math.max(0, Math.min(1, (0.59 - p[1]) / 0.325))
                  : 0;
          q[0] += 0.14 * (this.blade[0] / 640 - 0.5) * weight;
          q[2] += 0.12 * (this.blade[1] / 480 - 0.5) * weight;
          return q;
        });
        let v = points.map((p) => this.project(p));
        return { f, v, z: v.reduce((s, p) => s + p[2], 0) / v.length };
      })
      .sort((a, b) => b.z - a.z);
    for (let { f, v } of faces) {
      if (f.tag === "screen") {
        this.screen(f);
        continue;
      }
      let a = f.p[1].map((x, i) => x - f.p[0][i]),
        b = f.p[2].map((x, i) => x - f.p[0][i]),
        n = [
          a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0],
        ],
        norm = Math.hypot(...n),
        light = Math.abs(
          (-0.375 * n[0] + 0.797 * n[1] + 0.469 * n[2]) / (norm || 1),
        ),
        shade = 0.42 + 0.58 * light;
      c.fillStyle = `rgba(${f.c.map((x) => Math.round(x * shade)).join(",")},${f.a})`;
      c.beginPath();
      c.moveTo(v[0][0], v[0][1]);
      for (let p of v.slice(1)) c.lineTo(p[0], p[1]);
      c.closePath();
      c.fill();
      if (f.edge) {
        c.strokeStyle = f.tag === "wing" ? "#51788766" : "#591a1a99";
        c.lineWidth = 0.7;
        c.stroke();
      }
    }
    let p = this.project([
      1.05 + (this.blade[0] / 640 - 0.5) * 0.14,
      0.27,
      0.52 + (this.blade[1] / 480 - 0.5) * 0.12,
    ]);
    c.fillStyle = "#80ffdb";
    c.beginPath();
    c.arc(p[0], p[1], 3, 0, 7);
    c.fill();
    c.fillStyle = "#8295b1";
    c.font = "14px monospace";
    c.fillText("DRAG TO ORBIT  /  SCROLL TO ZOOM", 26, h - 24);
  }
  update(state) {
    this.frame = state.frame;
    this.blade = state.blade;
    this.draw();
  }
}
window.FlyDeskScene = FlyDeskScene;
