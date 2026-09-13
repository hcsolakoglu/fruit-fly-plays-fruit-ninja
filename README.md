# Fruit Fly Plays Fruit Ninja

**Tiny brain. Tiny sword.** A real MaleCNS connectome circuit controls a Fruit Ninja-style game through an explicit pixel → neural activity → blade loop.

4,386 real neurons · 468,923 measured edges · 4,210,625 synapses represented.

This is a reduced, fixed-connectivity rate model with engineered visual and motor interfaces. It is not a whole-brain emulation or evidence that a biological fly understands Fruit Ninja.

## Run the live experiment

Python 3.12+ recommended. No GPU is needed for playback or live inference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/serve.py
```

Open **http://127.0.0.1:8787** and press **Start experiment**. Switch among measured wiring, rewired wiring with its own fitted readout, connections cut, blinded input, random blade, and a direct pixel heuristic baseline. Seed controls reproducible launches. Simulation time is fixed at 30 steps/second; actual playback speed depends on CPU and browser latency.

The compact graph, fitted readouts, experiment outputs, traces and five finished videos are included. Raw 1.1 GB connectivity is not required to run.

## Watch the five edits

| Video | Angle | Format |
|---|---|---|
| [Tiny sword](videos/01_tiny_sword.mp4) | Four-shot gameplay montage at 1.5× | 1080 × 1920, 20s |
| [Pixels to neurons](videos/02_pixels_to_neurons.mp4) | Sensory field, neural computation, motor output | 1080 × 1920, 22s |
| [Cut the wiring](videos/03_cut_the_wiring.mp4) | Matched-seed causal ablation | 1920 × 1080, 20s |
| [Fruit vs bomb](videos/04_fruit_vs_bomb.mp4) | Actual hazard encounters and slow motion | 1080 × 1920, 20s |
| [Real vs rewired](videos/05_real_vs_rewired.mp4) | Control comparison and held-out results | 1080 × 1920, 24s |

Each video includes original synthesized music and event-timed effects. `videos/*.sources.json` maps every output frame to its saved source frame. Neural activity is sampled from the actual controller state; graph placement uses real soma coordinates. Video effects and camera movement are presentation only.

[Results and limitations](docs/RESULTS.md) · [Five X post concepts](docs/POSTS.md) · [Provenance](docs/SOURCES.md) · [Frozen experiment contract](docs/EXPERIMENT.md)

## What computes what

1. **Environment:** moving colored fruit and gray bombs, seeded launches, gravity, continuous blade-segment collisions. Blade speed is 700 px/s and its workspace is x=24..616, y=40..310; the bottom area is a launch zone. A hit removes the object and changes subsequent visible input.
2. **Engineered observation:** a flat-color RGB sensing pass of the game geometry, without cinematic shading, stems, HUD or effects, reduced to 96 × 64 pixels. Fixed colorfulness and achromatic-brightness channels are pooled to two 12 × 8 grids. This is palette-specific segmentation, not learned or biological vision.
3. **Engineered sensory injection:** seeded sparse random projection from those 192 values into 2,048 annotated visual projection neurons. This assignment does not claim real retinotopy.
4. **Connectome computation:** induced measured circuit with 1,024 central-brain intermediates and 1,314 descending neurons. Six updates `h = tanh(1.6 W h + input)` per observation, starting at zero. W contains signed, incoming-normalized measured synapse counts. Resetting between observations removes long-term neural memory; game-state feedback still closes the loop.
5. **Fitted readout:** ridge regression reconstructs the two spatial fields using **only descending activity**. No input neuron overlaps a readout neuron, and there is no pixel skip connection. A spatial peak selector and hazard/path suppression convert reconstructed fields to blade position and cutting state. Blade position is proprioceptive feedback.

Fitting the readout is supervised machine learning. No connectome weights are trained; no synaptic plasticity or biological adaptation is demonstrated. The neural controller never receives object labels, object coordinates, future trajectories or scores. Training targets are the same engineered fields extracted from visible synthetic scenes.

## Reproduce evidence and videos

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiment.py --dev
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/render_videos.py
```

FFmpeg with libx264 and DejaVu Sans fonts are needed for rendering. Rendering uses CPU and streams frames to FFmpeg. Development seeds 0–2 are used for recordings; the main montage uses all three, while the other videos use seed 1; evaluation uses 300–319 for six conditions and 400–409 for faster-motion intact runs. Every episode runs 600 frames (20 simulated seconds), with identical seeded launch schedules across conditions. Some objects remain in flight at the cutoff; the denominator includes every spawned fruit. Raw per-episode outcomes are in `results/evaluation.json`.

## Rebuild data and refit on Colab

Use the authenticated Google Colab CLI. L4 was used for the delivered model. Do not use a local GPU.

```bash
colab new -s fruit-fly-ninja --gpu L4
colab exec -s fruit-fly-ninja -f scripts/colab_prepare.py --timeout 600
colab exec -s fruit-fly-ninja -f scripts/build_connectome.py --timeout 300
tar -czf /tmp/fruit-code.tar.gz src scripts/train_readout.py
colab upload -s fruit-fly-ninja /tmp/fruit-code.tar.gz /content/fruit-code.tar.gz
colab exec -s fruit-fly-ninja -f scripts/colab_train.py --timeout 600
colab exec -s fruit-fly-ninja -f scripts/colab_pack.py --timeout 90
colab download -s fruit-fly-ninja /content/fruit-results.tar.gz /tmp/fruit-results.tar.gz
tar -xzf /tmp/fruit-results.tar.gz
colab stop -s fruit-fly-ninja
colab sessions
```

If any command fails, stop the allocated session after retrieving useful artifacts. The provided source manifest pins the downloaded data with hashes; verify them if rebuilding later. GPU sparse arithmetic may produce small differences across PyTorch/CUDA releases. Delivered readouts are the reproducible inference checkpoint.

Original code: MIT. MaleCNS derivative data: CC BY 4.0, credited in [SOURCES](docs/SOURCES.md).
