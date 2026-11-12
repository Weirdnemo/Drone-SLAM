# DroneForest

A drone version of the "Spot crossing procedural terrain" project: a quadrotor learns to fly through a
procedural forest with PPO, using a **frozen teacher + scan-gated imitation reward** and a
**per-env adaptive curriculum**, and a three.js viewer shows the flights with a live depth camera,
a voxel SLAM map and its surface mesh.

Everything runs on a laptop GPU (RTX 3050 mobile, 4 GB): the sim is a handful of torch tensors, no Isaac / Omniverse.

```
droneforest/
  terrain.py    analytic heightfield (same maths in torch and in the viewer's JS)
  dynamics.py   batched quadrotor, collective-thrust + body-rate actions, inner rate loop
  world.py      per-env forest (trunk cylinders + canopy spheres), batched raycasts, collisions
  env.py        obs (97-d), reward, curriculum, scan-gated imitation reward
  ppo.py        minimal PPO (own loop)
  heuristic.py  hand-written reference flyer (no training) - used for demo rollouts / sanity checks
train.py        teacher | forest | baseline   (--realism, --no-asym)
eval.py         benchmark a policy across forest densities (success, time, clearance, smoothness)
rollout.py      run a policy, export viewer/data/rollout.json (incl. INS drift + episode stats)
viewer/         three.js viewer (vendored, works offline)
```

## Run

```bash
pip install -r requirements.txt

# 1) open-field flyer = the frozen teacher
python train.py --stage teacher --iters 600 --envs 1024

# 2a) forest, initialised from the teacher + scan-gated imitation (the Spot trick)
python train.py --stage forest --teacher runs/teacher/model.pt --il-weight 0.5
# 2b) ablations
python train.py --stage forest --teacher runs/teacher/model.pt --il-weight 0   # init only
python train.py --stage baseline                                               # from scratch

# 3) export flights and look at them
python rollout.py --policy runs/forest/model.pt --level 0.8
cd viewer && python -m http.server 8000      # open http://localhost:8000
```

`python rollout.py --policy heuristic` makes demo flights without any training (a demo set is already in
`viewer/data/`). Training writes `runs/<stage>/log.csv` (reward, success, collision, curriculum level).
If you hit GPU memory limits lower `--envs` (512 is fine on 4 GB).

## How the Spot ideas map to the drone

| Spot post | here |
|---|---|
| frozen Isaac policy as teacher | frozen open-field flyer (stage 1) |
| scan-gated imitation reward | penalise `|a - a_teacher|^2` only where the range scan shows free space (`gate` in `env.py`); the teacher sees a ground-only scan, so it is never confused by trees |
| per-env adaptive curriculum | each env has its own forest density (`level`), +0.04 on success, -0.04 on crash |
| 45-cell height scan | 80-ray range scan (16 az x 5 el), analytic raycasts in torch |
| "policy ignores depth" | obstacles are placed on the start-goal corridor so the scan is required to succeed |

## Viewer

Controls: SPACE play/pause, R reset, N/P episode, F follow, mouse orbit/zoom, drag a `rollout.json` onto the window.
Panel: depth-camera range noise, point size, scan grid, SLAM mesh (rebuild on demand), fog density, sun shafts,
anisotropy, shadows, drone scale. The world (terrain + trees) is exactly what the policy flew in.
URL params: `?ep=2&t=8&noshadows=1`.

## Notes / honest limits

* The SLAM map is a voxel occupancy map built from the simulated depth camera with *ground-truth poses*
  (mapping, not full SLAM). The mesh is the exposed-face surface of occupied voxels.
* Sun shafts are a cheap additive-beam effect, not true volumetric lighting.
* The policy observes the 80-ray range scan, not depth images. A CNN on depth images is the natural next step.


## Research-informed upgrades (v2)

| Idea (source) | Where |
|---|---|
| Privileged learning: a critic / expert that sees clean ground truth while the deployed policy sees noisy sensors (Loquercio et al., *Learning high-speed flight in the wild*, Sci. Robotics 2021) | asymmetric critic: `PRIV_DIM` in `env.py`, `ActorCritic(critic_dim=...)` in `ppo.py` |
| Sensor + dynamics realism for sim-to-real: noisy ranges with dropouts, motor lag, drag, gusts (Aerial Gym Simulator 2025; Kaufmann et al., *Champion-level drone racing*, Nature 2023) | `env(realism=True)`: range noise growing with distance, 3 % dropouts, gyro/velocity noise, 1-step latency, OU wind gusts; per-env thrust/drag/time-constant randomisation |
| GPS-denied flight on inertial navigation (public material on Raphe mPhibr stresses in-house autopilots and INS) | position-estimate random-walk drift in the observation (`drift`), and in the viewer the map is built from the *drifting* pose |
| Compare against a from-scratch baseline and report density sweeps | `eval.py` |

Raphe mPhibr's public information is business-level (own autopilots/INS, swarm operation, AI detection, EW-adaptive
comms), so I used only the civil-relevant parts: noisy-state navigation and mapping. Swarm/multi-agent mapping is not implemented.

```bash
python train.py --stage teacher --realism
python train.py --stage forest --teacher runs/teacher/model.pt --realism --out runs/forest_il
python train.py --stage baseline --realism --out runs/baseline
python eval.py --policy runs/forest_il/model.pt --realism && python eval.py --policy runs/baseline/model.pt --realism
python rollout.py --policy runs/forest_il/model.pt --realism --drift 0.2 --device cuda
```

## Viewer: what the drone has mapped

* **Mapping scorecard** - *surface coverage* (share of the arena's true surfaces - terrain, trunks, canopies - that the map has captured),
  *map precision* (share of mapped voxels that sit on a true surface, +-1 voxel), *ground explored* (m^2), *INS drift now*.
* **Drifting-pose mapping** - the map is built from the drifting estimate, so with drift you see ghosted trees and precision fall;
  toggle it to compare with a perfect pose. `range noise` degrades precision the same way.
* **Dim unmapped ground** (fog-of-war on the terrain), **minimap** (explored cells, mapped obstacles in cyan, true trees in grey,
  true path in orange vs believed path in magenta, depth-camera FOV), **depth-camera image**, **telemetry + speed / nearest-return plot**.
* **Export map (.ply)** downloads the occupied voxels as a coloured point cloud.

## v3: disaster zone, procedural SLAM mission, new look

```bash
python mission.py --seed 11 --length 560          # long forest <-> disaster-zone flight -> viewer/data/mission.json
python rollout.py --scene disaster --policy heuristic --out viewer/data/disaster.json   # short arena episodes
python train.py --stage forest --teacher runs/teacher/model.pt --scene mixed            # train on forest + ruins
cd viewer && python -m http.server 8000     # / (mission)   ?data=disaster   ?data=rollout   ?cine=1   ?nopost=1
```

* **Sim**: `world.py` now has axis-aligned box obstacles (broken walls, rubble, fallen slabs, wrecked vehicles; slab-method raycast,
  collision, clearance) and `regenerate_disaster`. `--scene forest|disaster|mixed` in train / eval / rollout.
* **mission.py**: seeded world of alternating forest and ruin biomes along a wandering route; buildings, vehicles, logs, poles,
  stumps, fires, survivors/hikers. Seeds are retried until the flight completes. Flown by the heuristic flyer (or `--policy`) at a
  moving look-ahead waypoint, with wind and INS drift.
* **Viewer**: bloom / god rays / grade / vignette, ground-hugging mist, wind sway, grass, conifers, fires + smoke + embers with lights,
  chunked SLAM mesh that flashes in as it is mapped, scan pulse, FOV cone, fading cloud, survivor detection (line of sight, range, FOV)
  with pins and a mission overview strip, drone LEDs / fading trail / ground light, cinematic auto-camera (C) and WebM recording (V).
* **Honest limits**: the heuristic flyer, not a learned policy, flies the mission; the new box obstacles are not yet trained on
  (heuristic success on dense disaster arenas is modest). Contact-shadow decals replace screen-space AO. Box obstacles are axis-aligned.

## Asteroid proximity-ops mapping (SLAM)

```bash
python run_asteroid.py --kind all       # rubble pile, contact binary, elongated body; 6 runs each {perfect, dead-reckoned, SLAM} x {scripted, active}  (~5 min per body, CPU)
python run_asteroid.py --kind peanut --shape 7      # one body, another seed
python -m http.server 8000              # open http://localhost:8000/asteroid_viewer/   (?kind=peanut to pick a body; selector also in the GUI)
```
A spacecraft orbits a procedurally generated, spinning asteroid (analytic radius field, `asteroid/shape.py`; three body types with their own
gravity, spin and oblateness), scans it with a wide-FOV lidar and builds a lat/lon radial shape model over a six-phase mission (survey, mapping,
close proximity). Navigation drifts (random walk, 3 m/sqrt(h)); the SLAM mode registers each scan against the map built so far and corrects the
position. The active planner picks each orbit's inclination from the map's unobserved gaps. Tables per body: `runs/asteroid_results_<kind>.csv` / `.png`.

Viewer v2: sun-lit rocky surface (procedural albedo + bump, boulders, shadow-mapped craters), bloom/god-rays/grade, scan-reveal (mapped terrain
gains detail and flashes, unmapped terrain is a dark graticule), holographic model mesh coloured by error, live lidar footprint, true / believed /
dead-reckoned spacecraft ghosts with fading trails, crater and boulder pins with "mapped" toasts, lat/lon coverage map with ground track,
phase-labelled scrubbable mission strip, cinematic camera (C) and WebM recording (V, or "record whole run").
Idealisations: star-shaped (single-valued radius) shape model, point-mass + J2 gravity, impulsive plane changes, attitude and spin phase known,
a map-anchored correction rather than a full factor-graph SLAM. Boulders and surface texture are cosmetic (below the lidar's resolution).

## Lidar drone inside a Gaussian-splat scene

```bash
python run_splat.py --scene my_scan.ply --up=-y --scale 1.0   # 3DGS .ply or .splat; --up = which axis of the capture points up (COLMAP-style captures: -y)
python run_splat.py --demo                                    # stand-in procedural splat scene (NOT a real capture) to test the pipeline
python -m http.server 8000                                    # open http://localhost:8000/splat_viewer/   (?scene=<name>)
```
Splats have no hard geometry, so the sim needs a collision proxy: `splat/proxy.py` orients/scales the capture to a z-up metric frame (floor at z=0),
drops floaters, voxelises the opaque Gaussians (default 0.2 m), removes small blobs and takes a signed distance field. `splat/world.py` exposes that
SDF through the same interface as the procedural forest (collisions = SDF < drone radius, lidar = sphere tracing through the SDF), so the observation,
dynamics, noise/drift/wind/latency are unchanged. Start/goal pairs come from an A* path through the free space (`splat/plan.py`); the reference flyer
(`splat/follow.py`, privileged: it reads the SDF) follows that path. The viewer renders the *same transformed splats* with the Kellogg
GaussianSplats3D renderer (`splat_viewer/vendor/`, MIT) and replays the exported poses and 80-ray lidar ranges: live scan rays, a persistent SLAM point
map (switchable between corrected and drifting pose), trail, planned path, top-down minimap, several cameras, WebM recording.
Notes: scale and up-axis of a capture are arbitrary, check the printed extent is in metres and the grid looks sensible; the lidar hits the proxy,
not the Gaussians themselves, so it is accurate to about a voxel; the floor is a flat plane at the lowest 1% of the scene (indoor/uneven ground may
need `--floor`); the proxy is only as good as the splat's opacity (raise `--voxel`, lower `--thr` for sparse scans).

### Choosing start / end points, heights and collision safety
```
python run_splat.py --scene splat.ply --up=-y --scale 4 --fill-down --voxel 0.4 --start 12 -5 8 --goal 40 20 15        # exact points, z = metres above the floor
python run_splat.py ... --start 12 -5 3 --goal 40 20 3 --agl                                                          # z = metres above the local roof/canopy/ground
python run_splat.py ... --start-z 6 10 --goal-z 15 25 --ceiling 40                                                    # random pairs inside height bands
```
Click the **lidar heat map** in the viewer to get coordinates (click = start, shift+click = end; height = surface + the "pick height" slider) and paste the
printed `--start ... --goal ...`. Points inside geometry are moved to the nearest free spot (and reported); `--ceiling` lets the route fly over obstacles.
Collision safety: `--margin` (planner clearance, default 1.2 m), `--radius` (drone collision radius, 0.35 m), `--thr/--min-opacity` (lower = thicker proxy).
Every flight is also checked independently against the raw opaque splat centres: closer than `--splat-clear` (0.5 m) and it is rejected and not exported.
Viewer: dense accumulated **point cloud** + **voxel mesh** (size/opacity/hide-below sliders) from a 360° 90x10 mapping lidar, a top-down **heat map** of
mapped surface height and a live 360° **range image**; scan rays are off by default. Re-run `run_splat.py` to regenerate the rollout JSON.

## Safe-abort benchmark (GPS-denied forest mission)
```
python run_abort_bench.py --seeds 6          # 8 cases x 3 controllers x 6 seeds, ~8 min on CPU -> runs/abort_bench/{summary.md,summary.json,runs.csv,summary.png}
```
Mission: fly a forest corridor, hold over the target for 6 s (mapping), return and land on the pad, inside a geofence (x/y box + 20 m AGL). Each controller gets the same
world, start pose, route and noise stream. Controllers see only a drifting horizontal position estimate, a noisy 80-ray lidar, IMU attitude/velocity, an altimeter and a known
terrain map; the simulator scores with the true pose and clearance. Test cases change one thing at a time: `nominal`, `drift_low`/`drift_high`, `stale_short`/`stale_long`
(lidar frozen), `missing_8s` (no returns), `gap_open`/`gap_closed` (tree wall across the corridor with / without a passable gap). Cases below the trigger thresholds are
*nominal* (must finish; an abort is a false abort), the others are *abort-required* (must land safely on the pad within 40 s of the trigger).
Triggers (observables only, `Cfg` in `abortbench/core.py`): reported nav sigma > 1.5 m, no fresh lidar for 1.5 s, or blocked / no route progress for 3 s / 10 s.
Controllers: `baseline` (never aborts), `prescribed` (hold -> climb if overhead is clear, else retreat along the breadcrumb route if drift is small, else land -> land on the pad),
`chooser` (picks its own fallback from observables, no hold). Reported separately: completion, successful aborts, false aborts, collisions, geofence breaches, aborts that fail to
land on the pad in time, plus trigger latency / response / abort duration. Plug in your own controller by adding a branch to `run_case`.

### Interactive start / goal planning (no re-running run_splat.py)
```
python run_splat.py --scene splat.ply ... --name splat     # once, to build the scene + collision proxy
python plan_server.py                                      # serves the viewer AND plans/flies on demand
```
Open `http://localhost:8000/splat_viewer/`, then in the **Plan a flight** panel set *click scene to place* = start, click the splat in 3D (the pin lands on the surface under the cursor;
the goal is next), and set the start/goal heights above the surface with the sliders (or Q / E, Shift+Q / Shift+E). The route is planned live (cyan line, also on the heat map) and any point
inside geometry is moved to free space; messages tell you why a route is impossible (raise *max altitude*, lower *keep-out margin*). **Fly it (F)** runs the full simulation, with the
independent splat-distance check, and adds it as a new flight. Heat-map click / shift-click also sets start / goal. Without the server the viewer just prints the `--start ... --goal ...` command.

Viewer cameras: orbit, follow, chase, side, **birdseye** (top-down above the drone, north up, mouse wheel = height), **overview** (whole scene from above), cinematic (C cycles).
The **lidar map** (bottom left) is sized small / large / huge with M; it shows the mapped surface height, the live scan (white dots), the 20 m lidar range ring, the planned route and your start / goal.
