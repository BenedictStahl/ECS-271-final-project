"""Continue the approved 12-run grid, checking both GPUs every three minutes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/runs/finetuning-grid-2026-11-27"
INITIAL = ROOT / "data/runs/finetune-initial-2026-11-27-r2"
LOWER = ROOT / "data/runs/finetune-last2-lr3e-6-2026-11-27"


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(exist_ok=False)
    snapshots = OUT / "source_snapshot"
    snapshots.mkdir()
    for path in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(path, snapshots / path.name)
    shutil.copy2(INITIAL / "protocol.json", OUT / "protocol.json")
    base = read(INITIAL / "recipe.json")
    runs = []
    for key, depth, lr in [("A", 2, 1e-5), ("B", 2, 3e-6), ("C", 4, 1e-5), ("D", 4, 3e-6)]:
        config = dict(base, name=f"grid_{key}_last{depth}_lr{lr:g}", unfrozen_blocks=depth, encoder_lr=lr)
        config_path = OUT / f"recipe_{key}.json"
        config_path.write_text(json.dumps(config, indent=2))
        for model in ["deit3", "siglip2", "dinov3"]:
            adopted = key == "A" or (key == "B" and model == "deit3")
            folder = (INITIAL / model if key == "A" else LOWER / model) if adopted else OUT / f"{key}-{model}"
            runs.append(dict(configuration=key, model=model, config=str(config_path), output=str(folder),
                             adopted=adopted, status="pending"))
    processes = {}
    logs = {}
    environment = os.environ.copy()
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        environment[key] = "4"
    state = dict(started_utc=now(), poll_seconds=180, selection="seen_validation_auroc; no test or Qwen scoring",
                 trainer_sha256=hashlib.sha256((ROOT / "scripts/finetune.py").read_bytes()).hexdigest(), runs=runs)

    def persist():
        state["checked_utc"] = now()
        temporary = OUT / "grid_status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(OUT / "grid_status.json")
        lines = ["# Fine-tuning grid — 27 November 2026", "", "Approved: two encoder learning rates × two unfreezing depths × three encoders = 12 runs.",
                 "All other settings match the initial recipe. Same splits and seed; select using seen-validation AUROC only.",
                 "Checkpoints and predictions stay under data/runs. No test or Qwen evaluation is performed.",
                 "The scheduler checks progress and process exit codes every 180 seconds, with at most one training job per GPU.",
                 "Any failed job stops new launches; an already-running job is allowed to finish.", "",
                 f"Last check (UTC): {state['checked_utc']}", "", "| Config | Model | Blocks | Encoder LR | Status | Best val AUROC |", "|---|---|---:|---:|---|---:|"]
        for run in runs:
            config = read(Path(run["config"]))
            lines.append(f"| {run['configuration']} | {run['model']} | {config['unfrozen_blocks']} | {config['encoder_lr']:g} | {run['status']} | {run.get('best_validation_auroc', '')} |")
        lines += ["", "Live machine-readable record: data/runs/finetuning-grid-2026-11-27/grid_status.json.",
                  "Recipes, protocol, source snapshot, commands, timestamps, logs, and per-run histories are retained with the runs."]
        (ROOT / "docs/FINETUNING_GRID_2026-11-27.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    while True:
        occupied = set()
        for i, run in enumerate(runs):
            if run["adopted"]:
                status_path = INITIAL / "run_status.json" if run["configuration"] == "A" else LOWER / "run_status.json"
                external = read(status_path)
                entry = external["runs"][run["model"]] if "runs" in external else external
                run.update({k: v for k, v in entry.items() if k in ["status", "gpu", "pid", "exit_code", "started_utc", "finished_utc"]})
            elif i in processes:
                code = processes[i].poll()
                if code is not None and run["status"] == "running":
                    logs[i].close()
                    run.update(status="complete" if code == 0 else "failed", exit_code=code, finished_utc=now())
            folder = Path(run["output"])
            progress = folder / "progress.json"
            if progress.exists():
                run["progress"] = read(progress)
                if "best_validation_auroc" in run["progress"]:
                    run["best_validation_auroc"] = run["progress"]["best_validation_auroc"]
            if run["status"] == "running":
                occupied.add(run["gpu"])
                if progress.exists():
                    age = time.time() - progress.stat().st_mtime
                    if age > 900:
                        state["attention"] = f"No progress update for over 15 minutes: {run['configuration']}-{run['model']}"
        failed = any(run["status"] == "failed" for run in runs)
        if not failed and "attention" not in state:
            for gpu in [0, 1]:
                if gpu in occupied:
                    continue
                run = next((r for r in runs if not r["adopted"] and r["status"] == "pending"), None)
                if run is None:
                    continue
                command = [sys.executable, str(ROOT / "scripts/finetune.py"), "train", "--model", run["model"],
                           "--config", run["config"], "--protocol", str(OUT / "protocol.json"),
                           "--device", f"cuda:{gpu}", "--output", run["output"]]
                if run["model"] == "dinov3":
                    command += ["--local-dino", str((ROOT.parent / "dino_inspection_2026-11-25/dinov3-base").resolve())]
                i = runs.index(run)
                logs[i] = (OUT / f"{run['configuration']}-{run['model']}.log").open("w", encoding="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=logs[i], stderr=subprocess.STDOUT)
                processes[i] = process
                run.update(status="running", gpu=gpu, pid=process.pid, command=command, started_utc=now())
                print(f"Started {run['configuration']}-{run['model']} on GPU {gpu}; PID {process.pid}", flush=True)
        state["status"] = "needs_attention" if failed or "attention" in state else ("complete" if all(r["status"] == "complete" for r in runs) else "running")
        persist()
        print(now(), [(r["configuration"] + "-" + r["model"], r["status"]) for r in runs], flush=True)
        if state["status"] != "running":
            return
        time.sleep(180)


if __name__ == "__main__":
    main()
