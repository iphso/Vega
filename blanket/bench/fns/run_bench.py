"""Run one FNS benchmark model and store the per-bin detector rates as JSON.

  docker compose run --rm -v ~/runs/blanket/bench/fns:/fnsruns openmc \
      blanket/bench/fns/run_bench.py li2o --room --particles 1000000 --out /fnsruns/nominal/li2o_room.json
"""
import argparse, json, os, sys, time, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fnslib  # noqa: E402


def run(case, room=False, kinematic=False, particles=100_000, batches=20, seed=1, xs=None, work=None,
        threads=None, energy_tally=True, keep=False, full_sphere=False, room_air=True, fine_faces=False):
    model, info = fnslib.build(case, room=room, kinematic=kinematic, particles=particles // batches,
                               batches=batches, seed=seed, xs_xml=xs, energy_tally=energy_tally,
                               full_sphere=full_sphere, room_air=room_air, fine_faces=fine_faces)
    work = work or f"/fnsruns/work/{case}_{os.getpid()}"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    sp = model.run(cwd=work, threads=threads or int(os.environ.get("OMP_NUM_THREADS", "12")), output=False)
    wall = time.time() - t0
    res = fnslib.extract(sp, info)
    res["info"] = info
    res["info"].update(wall_s=wall, particles=particles)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["li2o", "be"])
    ap.add_argument("--room", action="store_true")
    ap.add_argument("--kin", action="store_true")
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--xs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--full-sphere", action="store_true", help="emit the backward source part too (no-room twin of a room run)")
    ap.add_argument("--fine-faces", action="store_true", help="1 mm tally bins within 1 cm of both faces")
    ap.add_argument("--void-room", action="store_true", help="room interior void (correlated room/no-room pair)")
    a = ap.parse_args()
    r = run(a.case, a.room, a.kin, a.particles, a.batches, a.seed, a.xs,
            work=os.path.splitext(a.out)[0] + "_work", keep=a.keep, full_sphere=a.full_sphere,
            room_air=not a.void_room, fine_faces=a.fine_faces)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w"))
    print(f"{a.case} room={a.room} kin={a.kin} N={a.particles} wall {r['info']['wall_s']:.0f}s -> {a.out}")
