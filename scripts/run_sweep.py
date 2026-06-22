"""Dispatch the cross-scene config sweep (Exp A/C/D) across N GPUs with a job queue.
Each job == one `exp_cross_scene.py` run pinned to one GPU. Logs to results/logs/<tag>.log.
"""
import subprocess, sys, time, os
from pathlib import Path

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
LOGS = REPO / "results" / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
PY = "/DATA/zihao/miniconda3/envs/mast3r_vggt/bin/python"

# (tag, extra args). Common: train 16 -> eval 4 unseen scenes.
JOBS = [
    # --- Exp A: main result + training depth ---
    ("A_rfon_frozen_s42",  "--method rf_on  --modality full --mode frozen  --steps 1800 --seed 42 --eval_controls"),
    ("A_rfoff_frozen_s42", "--method rf_off                 --mode frozen  --steps 1800 --seed 42"),
    ("A_rfon_frozen_s43",  "--method rf_on  --modality full --mode frozen  --steps 1800 --seed 43"),
    ("A_rfoff_frozen_s43", "--method rf_off                 --mode frozen  --steps 1800 --seed 43"),
    ("A_rfon_partial_s42", "--method rf_on  --modality full --mode partial --steps 2500 --seed 42 --eval_controls"),
    ("A_rfoff_partial_s42","--method rf_off                 --mode partial --steps 2500 --seed 42"),
    # --- Exp C: per-modality ablation (same arch, gate which RF inputs are real) ---
    ("C_paths",   "--method rf_on --modality paths   --mode frozen --steps 1800 --seed 42"),
    ("C_global",  "--method rf_on --modality global  --mode frozen --steps 1800 --seed 42"),
    ("C_angular", "--method rf_on --modality angular --mode frozen --steps 1800 --seed 42"),
    ("C_none",    "--method rf_on --modality none    --mode frozen --steps 1800 --seed 42"),
    # --- Exp D: angular conditioning encoder comparison ---
    ("D_cnn", "--method rf_on --modality full --mode frozen --steps 1800 --seed 42 --angular_encoder cnn"),
    ("D_vit", "--method rf_on --modality full --mode frozen --steps 1800 --seed 42 --angular_encoder shallow_vit"),
]


def main():
    gpus = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4,5,6,7").split(",")]
    queue = list(JOBS)
    running = {}  # gpu -> (proc, tag, t0)
    done = []
    while queue or running:
        for g in gpus:
            if g not in running and queue:
                tag, extra = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                log = open(LOGS / f"{tag}.log", "w")
                cmd = f"{PY} -u {REPO}/scripts/exp_cross_scene.py {extra} --tag {tag} --device cuda:0"
                p = subprocess.Popen(cmd.split(), env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
                running[g] = (p, tag, time.time(), log)
                print(f"[launch] gpu{g} <- {tag}", flush=True)
        for g, (p, tag, t0, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                dt = time.time() - t0
                ok = p.returncode == 0
                print(f"[{'done' if ok else 'FAIL'}] gpu{g} {tag} rc={p.returncode} ({dt:.0f}s)", flush=True)
                done.append((tag, p.returncode))
                del running[g]
        time.sleep(5)
    print("\n=== SWEEP COMPLETE ===")
    for tag, rc in done:
        print(f"  {tag}: rc={rc}")


if __name__ == "__main__":
    main()
