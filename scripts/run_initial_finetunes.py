"""Run two approved initial fine-tunes at once, queueing DINO on the first free GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread

from dataset_io import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-dino", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for source, target in (("finetuning_initial.json", "recipe.json"), ("training_protocol.json", "protocol.json")):
        (args.output / target).write_bytes((REPO_ROOT / source).read_bytes())
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(),
                  working_directory=str(REPO_ROOT), recipe=json.loads((args.output / "recipe.json").read_text()),
                  trainer_sha256=hashlib.sha256((REPO_ROOT / "scripts/finetune.py").read_bytes()).hexdigest(),
                  execution="One process per GPU; DeiT GPU0 + SigLIP GPU1 first, then DINO on first free GPU",
                  runs={name: dict(status="queued") for name in ("deit3", "siglip2", "dinov3")})
    lock = Lock()
    queue = ["dinov3"]
    failures = []

    def persist():
        temporary = args.output / "run_status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(args.output / "run_status.json")

    persist()
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "4"

    def worker(gpu, name):
        while name:
            command = [sys.executable, str(REPO_ROOT / "scripts/finetune.py"), "train", "--model", name,
                       "--config", str(args.output / "recipe.json"), "--protocol", str(args.output / "protocol.json"),
                       "--device", f"cuda:{gpu}", "--output", str(args.output / name)]
            if name == "dinov3":
                command += ["--local-dino", str(args.local_dino.resolve())]
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    with lock:
                        status["runs"][name] = dict(status="running", gpu=gpu, pid=process.pid,
                            started_utc=datetime.now(timezone.utc).isoformat(), command=command)
                        persist()
                    print(f"Started {name} on GPU {gpu}; PID {process.pid}", flush=True)
                    code = process.wait()
                except Exception as error:
                    code = -1
                    log.write(repr(error))
            with lock:
                status["runs"][name].update(status="complete" if code == 0 else "failed", exit_code=code,
                                          finished_utc=datetime.now(timezone.utc).isoformat())
                if code:
                    failures.append(name)
                print(f"Finished {name}: exit {code}", flush=True)
                name = queue.pop(0) if queue and not failures else None
                persist()

    threads = [Thread(target=worker, args=(0, "deit3")), Thread(target=worker, args=(1, "siglip2"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    status.update(status="failed" if failures else "complete", finished_utc=datetime.now(timezone.utc).isoformat())
    persist()
    if failures:
        raise SystemExit("A run failed; inspect logs. Further queued runs were not launched.")


if __name__ == "__main__":
    main()
