#!/usr/bin/env python3
"""Run bounded local inference batches while sampling Docker memory on Unraid."""
import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import time
import concurrent.futures
from benchmark_detection import request

def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=20).strip()

def snapshot(container):
    state = json.loads(docker("inspect", "--format", "{{json .State}}", container))
    row = json.loads(docker("stats", "--no-stream", "--format", "{{json .}}", container))
    return state, row

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", default="CodeProjectAI-Dev")
    p.add_argument("--url", default="http://127.0.0.1:32169")
    p.add_argument("--general", type=pathlib.Path, required=True)
    p.add_argument("--delivery", type=pathlib.Path, required=True)
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    if not (0 < a.hours <= 48 and a.interval >= 10 and 2 <= a.batch <= 100 and 1 <= a.workers <= 4):
        p.error("Invalid duration, interval, batch or worker count")
    images = {"IPcam-general": a.general.read_bytes(), "delivery": a.delivery.read_bytes()}
    def infer(model):
        return request(a.url, model, images[model], 0.4, 30)
    started = time.monotonic()
    total = failures = 0
    with a.output.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utc", "elapsed_s", "requests_total", "errors_total", "batch_mean_ms",
                         "memory_usage", "memory_percent", "cpu_percent", "pids", "started_at", "status", "oom_killed"])
        f.flush()
        initial_state, _ = snapshot(a.container)
        initial_start = initial_state["StartedAt"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            while time.monotonic() - started < a.hours * 3600:
                cycle = time.monotonic()
                jobs = [list(images)[i % 2] for i in range(a.batch)]
                rows = list(pool.map(infer, jobs))
                total += len(rows)
                failures += sum("error" in row for row in rows)
                good = [row["ms"] for row in rows if "error" not in row]
                state, stats = snapshot(a.container)
                writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    round(time.monotonic()-started, 1), total, failures,
                    round(statistics.mean(good), 2) if good else "",
                    stats["MemUsage"], stats["MemPerc"], stats["CPUPerc"], stats["PIDs"],
                    state["StartedAt"], state["Status"], state["OOMKilled"]])
                f.flush()
                print(f"Requests={total} failures={failures} memory={stats['MemUsage']}", flush=True)
                if state["StartedAt"] != initial_start or not state["Running"] or state["OOMKilled"]:
                    return 2
                if failures or float(stats["MemPerc"].rstrip("%")) >= 75:
                    print("Stopping at safety threshold or request failure.", flush=True)
                    return 1
                time.sleep(max(0, min(a.interval-(time.monotonic()-cycle), a.hours*3600-(time.monotonic()-started))))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
