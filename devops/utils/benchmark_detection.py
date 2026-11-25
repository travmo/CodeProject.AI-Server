#!/usr/bin/env python3
"""Local Blue Iris-compatible API benchmark; sends only user-selected images."""
import argparse
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
        return {"model": model, "ms": elapsed, "inferenceMs": result.get("inferenceMs"),
                "predictions": result.get("predictions", [])}
    except Exception as exc:
        return {"model": model, "error": str(exc), "ms": (time.perf_counter() - start) * 1000}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--image", type=pathlib.Path, required=True, help="Local JPEG test image")
    parser.add_argument("--requests", type=int, default=10, help="Requests per model")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    if args.requests < 1 or args.workers < 1 or args.timeout <= 0 or not 0 <= args.confidence <= 1:
        parser.error("Invalid request count, worker count, timeout or confidence")
    image = args.image.read_bytes()
    jobs = args.model * args.requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda model: request(args.url, model, image, args.confidence, args.timeout), jobs))
    summary = {}
    for model in args.model:
        rows = [r for r in results if r["model"] == model]
        times = sorted(r["ms"] for r in rows if "error" not in r)
        summary[model] = {"successes": len(times), "errors": len(rows)-len(times),
                          "median_ms": statistics.median(times) if times else None,
                          "p95_ms": times[math.ceil(.95*len(times))-1] if times else None,
                          "max_ms": max(times) if times else None}
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    return 1 if any("error" in r for r in results) else 0

if __name__ == "__main__":
    raise SystemExit(main())
