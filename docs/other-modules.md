# Other modules

## Asteroid proximity-ops mapping (SLAM)

```bash
python run_asteroid.py --kind all                   # rubble pile, contact binary, elongated body; 6 runs each {perfect, dead-reckoned, SLAM} x {scripted, active}  (~5 min per body, CPU)
python run_asteroid.py --kind peanut --shape 7      # one body, another seed
python -m http.server 8000                          # open http://localhost:8000/asteroid_viewer/   (?kind=peanut to pick a body; selector also in the GUI)
```

A spacecraft orbits a procedurally generated, spinning asteroid (analytic radius field, `asteroid/shape.py`; three body types with their own gravity, spin and oblateness), scans it with a wide-FOV lidar and builds a lat/lon radial shape model over a six-phase mission (survey, mapping, close proximity). Navigation drifts (random walk, 3 m/sqrt(h)); the SLAM mode registers each scan against the map built so far and corrects the position. The active planner picks each orbit's inclination from the map's unobserved gaps. Result tables per body: `runs/asteroid_results_<kind>.csv` and `.png`.

Viewer: sun-lit rocky surface (procedural albedo and bump, boulders, shadow-mapped craters), bloom / god rays / grade, scan-reveal (mapped terrain gains detail and flashes, unmapped terrain is a dark graticule), holographic model mesh coloured by error, live lidar footprint, true / believed / dead-reckoned spacecraft ghosts with fading trails, crater and boulder pins, lat/lon coverage map with ground track, phase-labelled scrubbable mission strip, cinematic camera (C) and WebM recording (V).

Idealisations: star-shaped (single-valued radius) shape model, point-mass + J2 gravity, impulsive plane changes, attitude and spin phase known, a map-anchored correction rather than a full factor-graph SLAM. Boulders and surface texture are cosmetic (below the lidar's resolution).
