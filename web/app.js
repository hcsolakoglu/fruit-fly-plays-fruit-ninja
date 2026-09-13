const $ = (id) => document.getElementById(id),
  colors = ["#fa7139", "#b55df9", "#3ee1b0", "#fbc442"];
let cfg,
  playing = false,
  busy = false,
  trail = [],
  effects = [],
  lastState = null;
const game = $("game").getContext("2d"),
  brain = $("brain").getContext("2d");
function heat(id, f) {
  let c = $(id).getContext("2d");
  c.fillStyle = "#0a1020";
  c.fillRect(0, 0, 240, 128);
  for (let y = 0; y < 8; y++)
    for (let x = 0; x < 12; x++) {
      let a = Math.min(1, Math.max(0, f[y * 12 + x]) * 3),
        b = Math.min(1, Math.max(0, f[96 + y * 12 + x]) * 3);
      c.fillStyle = `rgb(${Math.min(255, 14 + b * 232 + a * 26)},${20 + a * 215},${Math.min(255, 33 + a * 155 + b * 20)})`;
      c.fillRect(x * 20 + 1, y * 16 + 1, 18, 14);
    }
}
function draw(data) {
  let s = data.state;
  let baseline = ["pixel", "random"].includes(data.condition);
  $("brainlabel").textContent = baseline
    ? "02 / PASSIVE DISPLAY — BASELINE BYPASSES BRAIN"
    : "02 / LIVE NEURAL STATE";
  $("pipeline").textContent = baseline
    ? data.condition === "pixel"
      ? "BASELINE: visible pixels → engineered spatial controller → blade. Neural activity is not used."
      : "BASELINE: fixed seeded random targets → blade. Neural activity is not used."
    : "Visible pixels → visual neurons → circuit propagation → descending activity → fitted readout → blade";
  lastState = data;
  game.fillStyle = "#0b1222";
  game.fillRect(0, 0, 640, 480);
  game.strokeStyle = "#17253a";
  game.lineWidth = 1;
  for (let x = 0; x < 640; x += 80) {
    game.beginPath();
    game.moveTo(x, 0);
    game.lineTo(x, 480);
    game.stroke();
  }
  for (let y = 0; y < 480; y += 80) {
    game.beginPath();
    game.moveTo(0, y);
    game.lineTo(640, y);
    game.stroke();
  }
  game.fillStyle = "#0a0e19";
  game.fillRect(0, 310, 640, 170);
  game.strokeStyle = "#30445f";
  game.beginPath();
  game.moveTo(0, 310);
  game.lineTo(640, 310);
  game.stroke();
  game.font = "10px monospace";
  game.fillStyle = "#8191ab";
  game.fillText("LAUNCH ZONE / BLADE STAYS ABOVE", 18, 330);
  for (let o of s.objects) {
    game.save();
    game.translate(o.x, o.y);
    game.rotate(o.angle);
    let g = game.createRadialGradient(-o.r * 0.3, -o.r * 0.4, 1, 0, 0, o.r);
    g.addColorStop(0, o.bomb ? "#b9c3d8" : "#fff4cd");
    g.addColorStop(0.25, o.bomb ? "#73809e" : colors[o.kind]);
    g.addColorStop(1, o.bomb ? "#313c54" : colors[o.kind]);
    game.fillStyle = g;
    game.shadowColor = o.bomb ? "#ff779944" : colors[o.kind] + "55";
    game.shadowBlur = 12;
    game.beginPath();
    game.arc(0, 0, o.r, 0, Math.PI * 2);
    game.fill();
    game.shadowBlur = 0;
    if (o.bomb) {
      game.strokeStyle = "#f0b772";
      game.lineWidth = 3;
      game.beginPath();
      game.moveTo(0, -o.r);
      game.quadraticCurveTo(8, -o.r - 15, 18, -o.r - 8);
      game.stroke();
      game.fillStyle = "#e8ecf8";
      game.font = "bold 28px sans-serif";
      game.textAlign = "center";
      game.fillText("×", 0, 10);
    } else {
      game.fillStyle = "#69d797";
      game.beginPath();
      game.ellipse(5, -o.r, 11, 4, -0.4, 0, 7);
      game.fill();
    }
    game.restore();
  }
  for (let e of data.events) effects.push({ ...e, life: 22 });
  effects = effects.filter((e) => e.life-- > 0);
  for (let e of effects) {
    let dt = (22 - e.life) / 22;
    game.fillStyle = e.bomb ? "#ff5873" : colors[e.kind];
    for (let k = 0; k < 10; k++) {
      let a = k * 2.4 + e.id;
      game.beginPath();
      game.arc(
        e.x + Math.cos(a) * dt * 70,
        e.y + Math.sin(a) * dt * 70 + dt * dt * 30,
        Math.max(0.5, (1 - dt) * 4),
        0,
        7,
      );
      game.fill();
    }
  }
  trail.push({ xy: s.blade, on: s.slicing });
  trail = trail.slice(-10);
  game.lineCap = "round";
  for (let k = 1; k < trail.length; k++) {
    if (!trail[k].on) continue;
    game.strokeStyle = `rgba(106,245,203,${k / trail.length})`;
    game.lineWidth = k * 0.55;
    game.shadowColor = "#6af5cb";
    game.shadowBlur = 9;
    game.beginPath();
    game.moveTo(...trail[k - 1].xy);
    game.lineTo(...trail[k].xy);
    game.stroke();
  }
  game.shadowBlur = 0;
  game.strokeStyle = s.slicing ? "#f4fffa" : "#8494ad";
  game.lineWidth = 2;
  game.beginPath();
  game.arc(...s.blade, 6, 0, 7);
  game.stroke();
  $("score").textContent = `${s.sliced} FRUIT / ${s.bombs} BOMBS`;
  $("time").textContent = `${(s.frame / 30).toFixed(1)} SIMULATED SECONDS`;
  game.font = "bold 15px sans-serif";
  game.textAlign = "left";
  game.fillStyle = "#6af5cb";
  game.fillText(`${s.sliced} FRUIT`, 18, 27);
  game.textAlign = "right";
  game.fillStyle = "#ff7895";
  game.fillText(`${s.bombs} BOMBS`, 622, 27);
  game.textAlign = "left";
  brain.fillStyle = "#0e1627";
  brain.fillRect(0, 0, 600, 330);
  let theta = 0.05 * Math.sin(s.frame / 180),
    pos = cfg.coords.map((c) => [
      (c[0] * Math.cos(theta) + c[1] * Math.sin(theta) + 0.5) * 492 + 54,
      (c[2] * 0.66 - c[1] * 0.2 + 0.42) * 270 + 22,
    ]);
  let a = data.activity.map((v) => Math.min(1, Math.sqrt(Math.abs(v)) * 3));
  let edgePre = data.condition === "rewired" ? cfg.rewired_pre : cfg.pre,
    edgePost = data.condition === "rewired" ? cfg.rewired_post : cfg.post;
  if (data.condition === "disconnected") {
    edgePre = [];
    edgePost = [];
  }
  for (let k = 0; k < edgePre.length; k++) {
    let p = edgePre[k],
      q = edgePost[k],
      v = a[p] * 0.8 + a[q] * 0.2;
    brain.strokeStyle = `rgba(100,150,190,${0.08 + v * 0.55})`;
    brain.lineWidth = 0.7;
    brain.beginPath();
    brain.moveTo(...pos[p]);
    brain.lineTo(...pos[q]);
    brain.stroke();
  }
  for (let i = 0; i < a.length; i++) {
    brain.globalAlpha = 0.2 + 0.8 * a[i];
    brain.fillStyle =
      cfg.nodes[i].class === "visual_projection"
        ? "#6af5cb"
        : cfg.nodes[i].class === "descending_neuron"
          ? "#ffab70"
          : "#d283ff";
    brain.beginPath();
    brain.arc(...pos[i], 1 + 2.4 * a[i], 0, 7);
    brain.fill();
  }
  brain.globalAlpha = 1;
  heat("retina", data.retina);
  heat("field", data.field);
  $("sceneScore").textContent = `${s.sliced} FRUIT / ${s.bombs} BOMBS`;
  $("miniLabel").textContent = baseline
    ? "PASSIVE DISPLAY / BASELINE"
    : "LIVE CONNECTOME ACTIVITY";
  $("brain3d").getContext("2d").drawImage($("brain"), 0, 0, 320, 180);
  if (document.body.dataset.view === "3d") scene3d.update(s);
}
async function post(path, body = {}) {
  let r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw Error(`Server ${r.status}`);
  return r.json();
}
async function reset() {
  playing = false;
  $("toggle").textContent = "Start experiment ▶";
  trail = [];
  effects = [];
  try {
    draw(
      await post("api/reset", {
        condition: $("condition").value,
        seed: Number($("seed").value),
      }),
    );
    $("status").textContent = "Ready";
  } catch (e) {
    $("status").textContent = e.message;
  }
}
async function tick() {
  if (playing && !busy) {
    busy = true;
    let t = performance.now();
    try {
      draw(await post("api/step"));
      $("status").textContent =
        `${(performance.now() - t).toFixed(0)} ms / step`;
    } catch (e) {
      playing = false;
      $("status").textContent = e.message;
    }
    busy = false;
  }
  setTimeout(tick, 15);
}
$("toggle").onclick = () => {
  playing = !playing;
  $("toggle").textContent = playing ? "Pause Ⅱ" : "Resume ▶";
};
$("reset").onclick = reset;
$("condition").onchange = reset;
const scene3d = new FlyDeskScene($("scene3d"), $("game"));
$("mode3d").onclick = () => {
  document.body.dataset.view = "3d";
  $("mode3d").classList.add("primary");
  $("mode2d").classList.remove("primary");
  if (lastState) scene3d.update(lastState.state);
};
$("mode2d").onclick = () => {
  document.body.dataset.view = "2d";
  $("mode2d").classList.add("primary");
  $("mode3d").classList.remove("primary");
};
$("cameraReset").onclick = () => {
  scene3d.yaw = 0.5;
  scene3d.pitch = 0.37;
  scene3d.distance = 7.4;
  scene3d.draw();
};
const films = [
  ["06_fly_at_pc_3d", "A fly at a PC", "20s · 3D scene"],
  ["01_tiny_sword", "A tiny sword", "20s · spectacle"],
  ["02_pixels_to_neurons", "Pixels to neurons", "22s · inside the loop"],
  ["03_cut_the_wiring", "Cut the wiring", "20s · causal ablation"],
  ["04_fruit_vs_bomb", "Fruit vs bomb", "20s · errors in slow motion"],
  ["05_real_vs_rewired", "Real vs rewired", "24s · controlled comparison"],
];
$("filmgrid").innerHTML = films
  .map(
    ([n, t, s]) =>
      `<a class="film" href="videos/${n}.mp4"><img src="videos/${n}.jpg" alt="${t}" loading="lazy"><div>${t}<small>${s}</small></div></a>`,
  )
  .join("");
fetch("api/config")
  .then((r) => r.json())
  .then(async (c) => {
    cfg = c;
    await reset();
    tick();
  })
  .catch((e) => ($("status").textContent = e.message));
