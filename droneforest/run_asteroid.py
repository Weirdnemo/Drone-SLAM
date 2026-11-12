"""Asteroid proximity-ops mapping experiments.

  python run_asteroid.py --kind rubble          # 6 runs: {perfect, drift, slam} x {scripted, active}; writes viewer data + plot
  python run_asteroid.py --kind peanut --shape 7
  python run_asteroid.py --kind all             # rubble + peanut + elongated (writes data/index.json for the viewer's selector)
View:  python -m http.server 8000   ->  http://localhost:8000/asteroid_viewer/
"""
import argparse, csv, json, os, time
from asteroid.shape import Asteroid, KINDS
from asteroid.sim import run, PHASE_NAMES

DATA = "asteroid_viewer/data"


def do_kind(kind, a):
    ast = Asteroid(a.shape, kind); res, rows = {}, []
    print(f"=== {ast.name}  (seed {a.shape}) ===", flush=True)
    for pol in ("scripted", "active"):
        for mode in ("perfect", "drift", "slam"):
            t0 = time.time(); prim = (pol == "active" and mode == "slam")
            print(f"[{kind}: {pol:8s} / {mode:7s}]", flush=True)
            r = run(ast, mode, pol, hours_per_orbit=a.orbits, seed=a.seed, drift=a.drift, record=prim, log=print if prim else None)
            res[f"{pol}/{mode}"] = r["curve"]; c, rms, ve = r["final"]
            rows.append([pol, mode, round(100 * c, 1), round(rms, 2), round(ve, 2), r["curve"][-1][4], r["curve"][-1][5]])
            print(f"   final: coverage {100*c:.1f}%  RMS {rms:.1f} m  volume err {ve:+.1f}%  pose err {r['curve'][-1][4]:.1f} m  ({time.time()-t0:.0f}s)", flush=True)
            if prim: rec = r["rec"]
    os.makedirs("runs", exist_ok=True); os.makedirs(DATA, exist_ok=True)
    with open(f"runs/asteroid_results_{kind}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["policy", "mode", "coverage_pct", "rms_radial_err_m", "volume_err_pct", "final_pose_err_m", "nav_only_err_m"]); w.writerows(rows)
    out = {"asteroid": ast.params(), "grid": [45, 90], "frames": rec["frames"], "snaps": rec["snaps"], "snap_t": rec["snap_t"], "birth": rec["birth"], "phases": rec["phases"], "compare": res,
           "meta": {"mu": ast.mu, "fov_deg": 30, "drift_m_per_sqrt_h": a.drift * 1000, "primary": "active/slam", "sun": [1.0, -0.4, 0.5], "table": rows}}
    path = f"{DATA}/{kind}.json"; json.dump(out, open(path, "w"), separators=(",", ":"))
    print("wrote", path, f"{os.path.getsize(path)/1e6:.1f} MB", flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4)); col = {"perfect": "#2a9d4b", "drift": "#d1495b", "slam": "#2a6fdb"}
        for k, cv in res.items():
            pol, mode = k.split("/"); ls = "-" if pol == "active" else "--"
            for j, idx in enumerate((1, 2, 4)):
                ax[j].plot([c[0] for c in cv], [c[idx] * (100 if idx == 1 else 1) for c in cv], ls, color=col[mode], label=k)
        for j, tt in enumerate(["coverage (%)", "RMS radial error (m)", "position error (m)"]): ax[j].set_title(f"{ast.name}: {tt}"); ax[j].set_xlabel("mission time (h)"); ax[j].grid(alpha=.3)
        ax[1].set_ylim(0, 40); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"runs/asteroid_results_{kind}.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    return {"id": kind, "name": ast.name, "file": f"data/{kind}.json", "seed": a.shape}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="rubble", help="rubble | peanut | elongated | random | all")
    p.add_argument("--shape", type=int, default=3); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--drift", type=float, default=3e-3, help="nav random walk, km/sqrt(h)  (3e-3 = 3 m/sqrt(h))")
    p.add_argument("--orbits", type=float, default=1.0, help="orbits per phase (6 phases)")
    a = p.parse_args()
    kinds = list(KINDS) if a.kind == "all" else [a.kind]
    entries = [do_kind(k, a) for k in kinds]
    idx_path = f"{DATA}/index.json"
    old = json.load(open(idx_path)) if os.path.exists(idx_path) else []
    merged = {e["id"]: e for e in old}; merged.update({e["id"]: e for e in entries})
    order = [k for k in KINDS if k in merged] + [k for k in merged if k not in KINDS]
    json.dump([merged[k] for k in order], open(idx_path, "w"), indent=1); print("wrote", idx_path)


if __name__ == "__main__":
    main()
