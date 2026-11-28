"""Reproduce the five-map results and independently replay every solution.

Run: python submission/benchmark.py
No third-party dependencies. The unchanged course CLI is also usable:
python main.py --benchmark --map all
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepcore import heuristics, solver
from deepcore.environment import BUILTIN_MAPS, DeepCoreEnv


def validate(env, metrics):
    assert metrics is not None and metrics.solved, metrics
    state, cost = env.get_initial_state(), 0
    for action in metrics.solution_actions:
        assert action in env.get_legal_actions(state)
        state, step_cost = env.transition(state, action)
        cost += step_cost
    assert env.is_goal(state)
    assert cost == metrics.total_cost, (cost, metrics.total_cost)


def evaluate(env, label, fn):
    start = time.perf_counter()
    m = fn()
    elapsed = time.perf_counter() - start
    validate(env, m)
    row = asdict(m)
    row.pop("explored_order")
    row.update(algorithm=label, level=env.level_name.lower(), steps=len(m.solution_actions),
               runtime_sec=elapsed)
    print(f"{row['level']:6} {label:18} cost={m.total_cost:3} steps={row['steps']:3} "
          f"expanded={m.nodes_expanded:9,} frontier={m.max_frontier_size:7,} "
          f"memory={m.max_memory_size:9,} time={elapsed:.3f}s", flush=True)
    return row


def main():
    output = ROOT / "submission" / "benchmark.json"
    document = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "system": platform.platform(),
                "tie_breaking": "FIFO on equal f for A* H0/H1/H2; N,S,E,W successors",
                "timing": "Heuristic preprocessing is recorded separately; heuristic caches are warmed per map.",
                "memory_note": "Search-state counts exclude the mandatory diagnostic trace and heuristic preprocessing. IDS caches its graph; IDA* has a bounded transposition table. SMA*(unbounded) is its A* limit.",
                "rows": [], "preprocessing": [], "initial_heuristics": [],
                "heuristic_audit": [], "bounded_sma": []}

    def save():
        output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    for level in BUILTIN_MAPS:
        env = DeepCoreEnv.from_file_or_builtin(level)
        s0 = env.get_initial_state()
        t = time.perf_counter()
        initial = [h(s0, env) for h in (heuristics.h_naive_manhattan,
                                       heuristics.h1_basic, heuristics.h2_advanced)]
        abstract = heuristics._RELAXED_CACHE[env]
        document["preprocessing"].append({"level": level, "seconds": time.perf_counter() - t,
                                          "abstract_states": len(abstract[3]) if abstract else 0,
                                          "wall_distance_pairs": len(heuristics._WALL_DIST_CACHE[env])})
        document["initial_heuristics"].append({"level": level, "h0": initial[0], "h1": initial[1], "h2": initial[2]})
        suite = [
            ("BFS", lambda: solver.breadth_first_search(env)),
            ("DFS", lambda: solver.depth_first_search(env)),
            ("IDS", lambda: solver.iterative_deepening_search(env)),
            ("Bidirectional", lambda: solver.bidirectional_search(env)),
            ("UCS", lambda: solver.uniform_cost_search(env)),
            ("GBFS(H2)", lambda: solver.greedy_best_first_search(env, heuristics.h2_advanced)),
            ("A*(H0)", lambda: solver.a_star_search(env, heuristics.h_naive_manhattan)),
            ("A*(H1)", lambda: solver.a_star_search(env, heuristics.h1_basic)),
            ("A*(H2)", lambda: solver.a_star_search(env, heuristics.h2_advanced)),
            ("WA*(1.5)", lambda: solver.weighted_a_star_search(env, heuristics.h2_advanced, 1.5)),
            ("WA*(2.0)", lambda: solver.weighted_a_star_search(env, heuristics.h2_advanced, 2)),
            ("WA*(3.0)", lambda: solver.weighted_a_star_search(env, heuristics.h2_advanced, 3)),
            ("IDA*(H2)", lambda: solver.ida_star_search(env, heuristics.h2_advanced)),
            ("SMA*(unbounded)", lambda: solver.sma_star_search(env, heuristics.h2_advanced)),
        ]
        current_rows = {}
        for label, fn in suite:
            row = evaluate(env, label, fn)
            current_rows[label] = row
            document["rows"].append(row)
            save()
        optimal = current_rows["UCS"]["total_cost"]
        for label in ("A*(H0)", "A*(H1)", "A*(H2)", "IDA*(H2)", "SMA*(unbounded)"):
            assert current_rows[label]["total_cost"] == optimal
        for label in ("IDS", "Bidirectional"):
            assert current_rows[label]["steps"] == current_rows["BFS"]["steps"]
        assert 0 <= initial[0] <= initial[1] <= initial[2] <= optimal

        # Every suffix of a globally optimal path is itself optimal. This gives
        # exact residual costs at these states without trusting a heuristic.
        state, remaining, path_states = s0, optimal, 0
        for action in current_rows["A*(H2)"]["solution_actions"]:
            assert 0 <= heuristics.h1_basic(state, env) <= heuristics.h2_advanced(state, env) <= remaining
            state, cost = env.transition(state, action)
            remaining -= cost
            path_states += 1
        assert remaining == 0
        rng, state, edges = random.Random(20261128), s0, 0
        for _ in range(300):
            choices = env.get_successors(state)
            h0, h1, h2 = (h(state, env) for h in (heuristics.h_naive_manhattan,
                                                heuristics.h1_basic, heuristics.h2_advanced))
            assert 0 <= h0 <= h1 <= h2
            for _, child, cost in choices:
                assert h1 <= cost + heuristics.h1_basic(child, env)
                assert h2 <= cost + heuristics.h2_advanced(child, env)
                edges += 1
            state = rng.choice(choices)[1] if choices and rng.random() > 0.04 else s0
        document["heuristic_audit"].append({"level": level, "optimal_suffix_states": path_states,
                                              "sampled_edges": edges, "passed": True})
        save()

    env = DeepCoreEnv.from_file_or_builtin("level1")
    for cap in (32, 64, 128):
        row = evaluate(env, f"SMA*(cap={cap})",
                       lambda: solver.sma_star_search(env, heuristics.h2_advanced, cap))
        assert row["max_memory_size"] <= cap and row["total_cost"] == 68
        document["bounded_sma"].append(dict(row, memory_limit=cap))
        save()
    document["validated"] = True
    save()
    print(f"Validated {len(document['rows'])} main runs and {len(document['bounded_sma'])} bounded SMA* runs.", flush=True)


if __name__ == "__main__":
    main()
