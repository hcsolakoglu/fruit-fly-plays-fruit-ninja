# Results — protocol v2

The loop works, but this experiment does **not** show that biological wiring outperforms a rewired circuit or a simple pixel heuristic.

| Condition | Episodes | Fruit sliced / spawned | Mean episode hit rate | Bombs hit / spawned | Mean reward |
|---|---:|---:|---:|---:|---:|
| Measured MaleCNS wiring | 20 | 545 / 640 | 85.3% | 16 / 201 | 24.85 |
| Target-shuffled wiring + refitted readout | 20 | 544 / 640 | 85.2% | 16 / 201 | 24.80 |
| Connections cut | 20 | 0 / 640 | 0.0% | 0 / 201 | 0.00 |
| Blinded sensory input | 20 | 0 / 640 | 0.0% | 0 / 201 | 0.00 |
| Random blade | 20 | 90 / 640 | 13.9% | 28 / 201 | 0.30 |
| Engineered pixel heuristic | 20 | 544 / 640 | 85.1% | 15 / 201 | 24.95 |
| Measured wiring, faster motion | 10 | 287 / 324 | 88.6% | 7 / 111 | 26.60 |

Reward = sliced fruit − 3 × bomb hits. Standard conditions use matched seeds 300–319; faster motion uses 400–409. Each episode is 600 steps, 20 simulated seconds. All spawned fruit count in the denominator, including fruit still in flight at cutoff. There are no selected or excluded final episodes.

The paired real-minus-rewired reward difference is 0.05 per episode. A descriptive 10,000-resample paired bootstrap 95% interval is [-0.30, 0.40]. This is not a broad biological comparison: only one circuit extraction, one sensory mapping and one shuffled graph were tested.

## What worked

- Actual visible RGB drives the neural circuit. The fitted readout receives only descending-neuron rates; changes in neural output move the blade. Slicing removes fruit and changes the next observation.
- All 468,923 connections in the intact reservoir correspond to published MaleCNS neuron pairs. Relative incoming synapse counts are retained, with engineered normalization and coarse transmitter signs.
- The disconnected and blinded interventions eliminate scoring on the same launches. This supports causal dependence of this controller on sensory drive and connectome propagation.
- The controller works on fresh seeds. Faster-motion episodes also run successfully; their higher fruit score may reflect fruit spending more time inside the blade workspace, not improved reaction ability.
- Five main videos plus a 3D bonus use actual development traces (seeds 0–2 for the main montage; seed 1 for the others), not the final evaluation episodes. Their quantitative comparison card uses all final evaluation episodes. Event particles and sound are editorial effects tied to recorded collisions.

## What did not work / what is not established

- Real wiring did not meaningfully outperform the rewired control or engineered pixel baseline.
- Bomb avoidance is imperfect. Coarse spatial cells, selecting one peak, and straight blade movement can create collisions even when the brightness channel represents a bomb.
- No reinforcement learning, online adaptation, synaptic plasticity or biological learning was demonstrated. The only learning is supervised ridge fitting of an engineered readout on 6,500 generated still-image scenes.
- Readout validation on 600 separate generated scenes gave field MSE 0.00001247 for intact wiring and 0.00000355 for rewired wiring, versus 0.00262548 for an all-zero field. These are reconstruction errors, not game rewards or fly-learning measurements.
- The controller sees a simplified flat-color RGB sensing pass of the same object geometry, without cinematic shading, stems, HUD or particles. Vision is deliberately engineered: fixed colorfulness and gray-brightness segmentation, pooled to two 12 × 8 fields. The neural model does not discover fruit/bomb categories from raw images. Unseen palettes, textures or real-world scenes were not tested.

## Modeling boundaries

This is a 4,386-neuron induced subgraph: 2,048 visual projection neurons, 1,024 central-brain bridge neurons and 1,314 descending neurons. It represents 4,210,625 counted synapses. It is not a complete MaleCNS brain or a biophysically calibrated simulator.

Six tanh propagation steps start from zero on each observation. There is no persistent neural memory, explicit spike timing, or biological latency claim. Rate values can be signed; visualization brightness uses absolute activity, with a fixed square-root display scale. Soma positions and a selected subset of real edges are shown, not reconstructed axon shapes.

The sensory projection is a fixed seeded random assignment rather than anatomical retinotopy. Descending neurons are biologically motivated output locations, but their game coordinate meanings and readout are engineered. GABA/glutamate are negative; acetylcholine, modulatory and unclear predictions are positive in this simplified model. NT predictions and source reconstruction are uncertain.

The shuffled control permutes target stubs, retaining source identity and edge-weight multiset before sparse duplicate merging, then renormalizes incoming absolute weights. It is refitted on the identical training scenes with the same ridge penalty. It is not a perfect preservation of every graph statistic. The random baseline uses one fixed random-action stream across environment seeds.

## Development change and limits of the holdout

Protocol v1 allowed slicing immediately at the launch edge and produced near-perfect harvesting. Visual review of the development trace showed a trivial, visually empty arena. Protocol v2 reduced blade speed to 700 pixels/second and restricted its workspace above y=310. All conditions were rerun. V1 results remain in `results/protocol_v1`; new final seeds were locked before v2 evaluation, and no model parameters were tuned on their results. The holdout was procedural, not access-controlled.

## Reproduction and verification

Per-episode rows: `results/evaluation.json`. Aggregate values: `results/evaluation_summary.json`. Training evidence: `results/training.json`. Reproducible recordings: `results/recordings`. Frame lineage: `videos/*.sources.json`. Integrity tests check real-edge preservation, sensory-to-output influence, zero descending signal after disconnection, deterministic closed-loop behavior, and matched launch schedules.

The 3D stage is an orbitable fly-at-PC mesh with the actual game trace on the monitor. Its body geometry, mouse/foreleg motion and camera are engineered presentation; this is not a biological fly-body simulation. The neural inset remains actual recorded activity.

The visual trace at index f contains the rates computed from the observation before step f; its displayed game snapshot is the resulting state after that action. This is the one-step causal ordering, not a fabricated simultaneous measurement.

Compute: one Colab NVIDIA L4 session, explicitly terminated after artifact retrieval. The successful fitting pass took approximately 19.6 seconds including synthetic scene construction and GPU solves. The CLI did not provide reliable exact credit consumption. No local GPU was used. Local CPU handled preprocessing inspection, evaluation, rendering, and the live server.
