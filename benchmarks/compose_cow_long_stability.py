"""
Long stability benchmark for Gecko Compose
- Default: 30 minutes run, 1000 nodes (50 layers x 20 nodes), history_depth=200
- Records per-run metrics into `/workspaces/gecko/benchmarks/results_long_stability.json`
- After completion, compares with `results_cow_performance.json` if present and writes a summary report.
"""
import asyncio
import time
import psutil
import os
import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict

from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import WorkflowContext

RUN_DURATION_SECONDS = int(os.environ.get("DURATION", "1800"))  # default 30 minutes
ITERATION_SLEEP = float(os.environ.get("ITERATION_SLEEP", "1.0"))
HISTORY_DEPTH = int(os.environ.get("HISTORY_DEPTH", "200"))
LAYERS = int(os.environ.get("LAYERS", "50"))
NODES_PER_LAYER = int(os.environ.get("NODES_PER_LAYER", "20"))
OUTPUT_JSON = "/workspaces/gecko/benchmarks/results_long_stability.json"
REPORT_MD = "/workspaces/gecko/benchmarks/RESULTS_LONG_STABILITY_REPORT.md"
PREV_RESULTS = "/workspaces/gecko/benchmarks/results_cow_performance.json"


def create_computation_dag(num_layers: int, nodes_per_layer: int) -> Workflow:
    engine = Workflow(name="LongStabilityDAG", max_steps=num_layers + 10)
    engine.graph.add_node("entry", lambda ctx: {"sum": 0})
    engine.graph.set_entry_point("entry")

    prev_layer = ["entry"]
    for layer_idx in range(1, num_layers + 1):
        layer_nodes = []
        for node_idx in range(nodes_per_layer):
            node_name = f"L{layer_idx}_N{node_idx}"
            def make_node(l, n):
                def node_func(ctx: WorkflowContext):
                    # small but numerous state mutations
                    for i in range(10):
                        ctx.state[f"t_{l}_{n}_{i}"] = l * n * i
                    # produce a small dict
                    return {"layer": l, "node": n, "ok": True}
                return node_func
            engine.graph.add_node(node_name, make_node(layer_idx, node_idx))
            layer_nodes.append(node_name)
        # wire: depend on last of prev layer
        if prev_layer:
            for node in layer_nodes:
                engine.graph.add_edge(prev_layer[-1], node)
        prev_layer = layer_nodes

    return engine


def add_historical_load(ctx: WorkflowContext, depth: int):
    for i in range(depth):
        ctx.history[f"step_{i}"] = {"timestamp": time.time(), "data": list(range(50))}


async def run_once(engine: Workflow, history_depth: int) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / (1024 * 1024)
    ctx = WorkflowContext(input={})
    if history_depth > 0:
        add_historical_load(ctx, history_depth)
    start = time.time()
    try:
        await engine.execute(input_data=None, _resume_context=ctx, timeout=300.0)
        status = "OK"
        err = None
    except Exception as e:
        status = "FAILED"
        err = str(e)
    end = time.time()
    peak = process.memory_info().rss / (1024 * 1024)
    return {
        "timestamp": datetime.now().isoformat(),
        "duration_s": end - start,
        "baseline_mb": baseline,
        "peak_mb": peak,
        "delta_mb": peak - baseline,
        "status": status,
        "error": err,
        "total_nodes": len(engine.graph.nodes)
    }


async def main():
    print("Starting long stability benchmark")
    engine = create_computation_dag(LAYERS, NODES_PER_LAYER)
    total_nodes = len(engine.graph.nodes)
    print(f"DAG built: layers={LAYERS}, nodes_per_layer={NODES_PER_LAYER}, total_nodes={total_nodes}")
    end_time = datetime.now() + timedelta(seconds=RUN_DURATION_SECONDS)

    results = []
    iteration = 0
    while datetime.now() < end_time:
        iteration += 1
        print(f"Run #{iteration} at {datetime.now().isoformat()}")
        rec = await run_once(engine, HISTORY_DEPTH)
        results.append(rec)
        # persist intermediate results
        with open(OUTPUT_JSON, "w") as f:
            json.dump({"started_at": (datetime.now() - timedelta(seconds=RUN_DURATION_SECONDS)).isoformat(), "iteration": iteration, "results": results}, f, indent=2)
        # small sleep to stabilize
        await asyncio.sleep(ITERATION_SLEEP)

    # Final save
    meta = {
        "completed_at": datetime.now().isoformat(),
        "run_duration_s": RUN_DURATION_SECONDS,
        "layers": LAYERS,
        "nodes_per_layer": NODES_PER_LAYER,
        "history_depth": HISTORY_DEPTH,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Saved stability results to {OUTPUT_JSON}")

    # compare with previous
    summary_lines = []
    if os.path.exists(PREV_RESULTS):
        try:
            with open(PREV_RESULTS) as f:
                prev = json.load(f)
            # compute simple aggregates
            prev_ok = [r for r in prev.get("results", []) if r.get("status") == "OK"]
            curr_ok = [r for r in results if r.get("status") == "OK"]
            prev_avg_time = sum(r.get("execution_time_ms", r.get("duration_s", 0)*1000) for r in prev_ok)/max(1, len(prev_ok))
            curr_avg_time = sum(r.get("duration_s", 0) for r in curr_ok)/max(1, len(curr_ok)) * 1000
            prev_avg_mem = sum(r.get("memory_delta_mb", r.get("delta_mb", 0)) for r in prev_ok)/max(1, len(prev_ok))
            curr_avg_mem = sum(r.get("delta_mb", 0) for r in curr_ok)/max(1, len(curr_ok))

            summary_lines.append(f"Previous avg time (ms): {prev_avg_time:.1f}")
            summary_lines.append(f"Current avg time (ms): {curr_avg_time:.1f}")
            summary_lines.append(f"Previous avg mem delta (MB): {prev_avg_mem:.2f}")
            summary_lines.append(f"Current avg mem delta (MB): {curr_avg_mem:.2f}")
        except Exception as e:
            summary_lines.append(f"Failed to compare with previous results: {e}")
    else:
        summary_lines.append("No previous baseline results found for comparison.")

    with open(REPORT_MD, "w") as f:
        f.write("# Long Stability Benchmark Report\n\n")
        f.write("## Meta\n")
        f.write(json.dumps(meta, indent=2))
        f.write("\n\n## Summary\n")
        for l in summary_lines:
            f.write(f"- {l}\n")
        f.write("\n## Runs\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Report written to {REPORT_MD}")


if __name__ == '__main__':
    asyncio.run(main())
