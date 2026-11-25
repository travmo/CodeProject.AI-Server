#!/usr/bin/env python3
"""Local Blue Iris-compatible API benchmark; sends only user-selected images."""
import argparse
import collections
import concurrent.futures
import json
import math
import pathlib
import statistics
import time
import urllib.request
import urllib.parse
import uuid

def request(base, model, image, confidence, timeout):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"min_confidence\"\r\n\r\n{confidence}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.jpg\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + image + f"\r\n--{boundary}--\r\n".encode()
    url = base.rstrip("/") + "/v1/vision/custom/" + urllib.parse.quote(model, safe="")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - start) * 1000
        if result.get("success") is not True:
            return {"model": model, "error": str(result.get("error") or result.get("message") or "API failure"), "ms": elapsed}
        return {"model": model, "ms": elapsed, "inferenceMs": result.get("inferenceMs"), "processMs": result.get("processMs"),
                "predictions": result.get("predictions", [])}
    except Exception as exc:
        return {"model": model, "error": str(exc), "ms": (time.perf_counter() - start) * 1000}

def metrics(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p95": round(values[math.ceil(.95 * len(values)) - 1], 2),
            "max": round(max(values), 2)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.1.43:32169")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--image", type=pathlib.Path, action="append", required=True,
                        help="One JPEG per model, or one shared JPEG")
    parser.add_argument("--requests", type=int, default=20, help="Requests per model per phase")
    parser.add_argument("--workers", type=int, action="append", help="Concurrency per phase; default 1, 2, 4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=pathlib.Path, help="Optional local JSON report")
    args = parser.parse_args()
    workers = args.workers or [1, 2, 4]
    if (args.requests < 1 or min(workers) < 1 or max(workers) > 8 or
        args.warmup < 0 or args.timeout <= 0 or not 0 <= args.confidence <= 1 or
        len(set(args.model)) != len(args.model) or len(args.image) not in (1, len(args.model))):
        parser.error("Invalid counts, duplicate models, image count, confidence or timeout")
    images = [p.read_bytes() for p in args.image]
    model_images = dict(zip(args.model, images if len(images) > 1 else images * len(args.model)))
    def run(model):
        return request(args.url, model, model_images[model], args.confidence, args.timeout)
    def labels(row):
        return tuple(sorted(str(p.get("label", "")) for p in row.get("predictions", [])))
    baseline = {}
    for model in args.model:
        for _ in range(args.warmup):
            row = run(model)
            if "error" in row:
                print(json.dumps({"warmup_error": row}, indent=2))
                return 1
            baseline[model] = labels(row)
    report = {"url": args.url, "models": args.model, "requests_per_model": args.requests,
              "warmup_per_model": args.warmup, "confidence": args.confidence, "phases": []}
    failed = False
    for count in workers:
        jobs = args.model * args.requests
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(run, jobs))
        duration = time.perf_counter() - start
        good = [r for r in results if "error" not in r]
        phase = {"workers": count, "elapsed_seconds": round(duration, 3),
                 "successful_requests_per_second": round(len(good)/duration, 2), "models": {}}
        for model in args.model:
            rows = [r for r in results if r["model"] == model]
            ok = [r for r in rows if "error" not in r]
            errors = collections.Counter(r["error"] for r in rows if "error" in r)
            failed |= bool(errors)
            phase["models"][model] = {
                "successes": len(ok), "errors": dict(errors),
                "roundtrip_ms": metrics([r["ms"] for r in ok]),
                "process_ms": metrics([r.get("processMs") for r in ok]),
                "inference_ms": metrics([r.get("inferenceMs") for r in ok]),
                "label_counts_match_warmup": sum(labels(r) == baseline[model] for r in ok) if model in baseline else None,
                "label_patterns": dict(collections.Counter(", ".join(labels(r)) or "(none)" for r in ok))}
        report["phases"].append(phase)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
