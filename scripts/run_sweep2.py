"""Second wave: partial-unfreeze (2500-step) re-runs of the modality ablation + a partial
seed-43 replicate, so the ablation & headline use the well-converged training regime."""
import subprocess, sys, time, os
from pathlib import Path

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
LOGS = REPO / "results" / "logs"
PY = "/DATA/zihao/miniconda3/envs/mast3r_vggt/bin/python"

JOBS = [
    ("C2_paths",   "--method rf_on --modality paths   --mode partial --steps 2500 --seed 42"),
    ("C2_global",  "--method rf_on --modality global  --mode partial --steps 2500 --seed 42"),
    ("C2_angular", "--method rf_on --modality angular --mode partial --steps 2500 --seed 42"),
    ("C2_none",    "--method rf_on --modality none    --mode partial --steps 2500 --seed 42"),
    ("A_rfon_partial_s43",  "--method rf_on  --modality full --mode partial --steps 2500 --seed 43 --eval_controls"),
    ("A_rfoff_partial_s43", "--method rf_off                 --mode partial --steps 2500 --seed 43"),
]


def main():
    gpus = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4,5,6").split(",")]
    queue = list(JOBS); running = {}; done = []
    while queue or running:
        for g in gpus:
            if g not in running and queue:
                tag, extra = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                log = open(LOGS / f"{tag}.log", "w")
                cmd = f"{PY} -u {REPO}/scripts/exp_cross_scene.py {extra} --tag {tag} --device cuda:0"
                p = subprocess.Popen(cmd.split(), env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
                running[g] = (p, tag, time.time(), log); print(f"[launch] gpu{g} <- {tag}", flush=True)
        for g, (p, tag, t0, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); print(f"[{'done' if p.returncode==0 else 'FAIL'}] gpu{g} {tag} rc={p.returncode} ({time.time()-t0:.0f}s)", flush=True)
                done.append((tag, p.returncode)); del running[g]
        time.sleep(5)
    print("\n=== SWEEP2 COMPLETE ===")
    for tag, rc in done: print(f"  {tag}: rc={rc}")


if __name__ == "__main__":
    main()
