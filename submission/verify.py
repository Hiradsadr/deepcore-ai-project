"""Additional independent checks; the supplied tests are never modified.

Run from the repository root: python submission/verify.py
Uses only Python's standard library. Small maps are exhaustively enumerated;
exact distances are computed from their actual forward edges, not the solver.
"""

from collections import defaultdict, deque
from dataclasses import replace
import heapq
from itertools import count
import math
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepcore import heuristics, solver
from deepcore.environment import DeepCoreEnv


def replay(env, result):
    state, cost = env.get_initial_state(), 0
    for action in result.solution_actions:
        if action not in env.get_legal_actions(state):
            raise AssertionError(f"Illegal action {action}")
        state, step = env.transition(state, action)
        cost += step
    if result.solved:
        assert env.is_goal(state), "Solution did not reach a goal"
        assert result.total_cost == cost, "Reported and replayed costs differ"
    return cost


SMALL_MAPS = {
    "plain": "#######\n#S.M.E#\n#######",
    "rubble": "#######\n#S@M.E#\n#######",
    "bridge": "########\n#SBOM.E#\n########",
    "unused-pit": "########\n#S.M..E#\n###O####\n########",
    "unusable-stone": "########\n#B#..ME#\n#S..O..#\n########",
    "two-miners": "########\n#S.~M.E#\n#M.....#\n########",
    "blocked": "#######\n#S#M.E#\n#######",
    "optional-rubble": "#######\n#S.M.E#\n#..@..#\n#######",
    "no-miners": "#####\n#S.E#\n#####",
}

# Reproducible small caves with interacting terrain, miners and movable stones.
_random = random.Random(20261128)
for _case in range(16):
    _grid = [["#"] * 7 for _ in range(5)]
    for _r in range(1, 4):
        for _c in range(1, 6):
            _grid[_r][_c] = "#" if _random.random() < 0.15 else "."
    _grid[1][1], _grid[3][5] = "S", "E"
    _places = [(r, c) for r in range(1, 4) for c in range(1, 6)
               if (r, c) not in ((1, 1), (3, 5))]
    _random.shuffle(_places)
    for (_r, _c), _symbol in zip(_places, "MMBO@~"):
        _grid[_r][_c] = _symbol
    SMALL_MAPS[f"seeded-{_case:02}"] = "\n".join("".join(row) for row in _grid)


def exact_graph(env):
    start = env.get_initial_state()
    states, queue, reverse = {start}, deque([start]), defaultdict(list)
    edges = []
    while queue:
        state = queue.popleft()
        for _, child, cost in env.get_successors(state):
            edges.append((state, child, cost))
            reverse[child].append((state, cost))
            if child not in states:
                states.add(child)
                queue.append(child)
    distances, heap, serial = {}, [], count()
    for state in states:
        if env.is_goal(state):
            distances[state] = 0
            heapq.heappush(heap, (0, next(serial), state))
    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost != distances[state]:
            continue
        for previous, step in reverse[state]:
            new = cost + step
            if new < distances.get(previous, math.inf):
                distances[previous] = new
                heapq.heappush(heap, (new, next(serial), previous))
    return states, edges, distances


class SubmissionChecks(unittest.TestCase):
    checked_states = 0
    checked_edges = 0

    def test_exhaustive_heuristic_properties(self):
        for name, text in SMALL_MAPS.items():
            with self.subTest(map=name):
                env = DeepCoreEnv(text, name)
                states, edges, exact = exact_graph(env)
                values = {}
                for state in states:
                    h0 = heuristics.h_naive_manhattan(state, env)
                    h1 = heuristics.h1_basic(state, env)
                    h2 = heuristics.h2_advanced(state, env)
                    self.assertTrue(0 <= h0 <= h1 <= h2 <= exact.get(state, math.inf),
                                    (name, state, h0, h1, h2, exact.get(state)))
                    if env.is_goal(state):
                        self.assertEqual((h0, h1, h2), (0, 0, 0))
                    values[state] = (h1, h2)
                for state, child, cost in edges:
                    for a, b in zip(values[state], values[child]):
                        self.assertLessEqual(a, cost + b, (name, state, child, a, b, cost))
                type(self).checked_states += len(states)
                type(self).checked_edges += len(edges)

    def test_algorithms_against_independent_oracle(self):
        for name, text in SMALL_MAPS.items():
            with self.subTest(map=name):
                env = DeepCoreEnv(text, name)
                _, _, exact = exact_graph(env)
                optimal = exact.get(env.get_initial_state(), math.inf)
                bfs = solver.breadth_first_search(env)
                for fn in (solver.depth_first_search, solver.iterative_deepening_search,
                           solver.bidirectional_search, solver.uniform_cost_search):
                    result = fn(env)
                    self.assertEqual(result.solved, math.isfinite(optimal))
                    replay(env, result)
                    if result.solved and fn in (solver.iterative_deepening_search, solver.bidirectional_search):
                        self.assertEqual(len(result.solution_actions), len(bfs.solution_actions))
                    if result.solved and fn is solver.uniform_cost_search:
                        self.assertEqual(result.total_cost, optimal)
                for h in (heuristics.h_naive_manhattan, heuristics.h1_basic, heuristics.h2_advanced):
                    for fn in (solver.a_star_search, solver.ida_star_search):
                        result = fn(env, h)
                        self.assertEqual(result.solved, math.isfinite(optimal))
                        replay(env, result)
                        if result.solved:
                            self.assertEqual(result.total_cost, optimal)
                bounded = solver.sma_star_search(env, heuristics.h2_advanced, 64)
                replay(env, bounded)
                self.assertEqual(bounded.solved, math.isfinite(optimal))
                self.assertLessEqual(bounded.max_memory_size, 64)
                if bounded.solved:
                    self.assertEqual(bounded.total_cost, optimal)

    def test_reopen_with_inconsistent_admissible_heuristic(self):
        # S->A is expanded first; S->B->A then improves an already closed A.
        template = DeepCoreEnv(SMALL_MAPS["plain"]).get_initial_state()
        states = [replace(template, bot_pos=(0, i)) for i in range(4)]
        edges = {states[0]: [("SA", states[1], 3), ("SB", states[2], 1)],
                 states[1]: [("AG", states[3], 3)],
                 states[2]: [("BA", states[1], 1), ("BG", states[3], 100)], states[3]: []}

        class Graph:
            def get_initial_state(self):
                return states[0]

            def is_goal(self, state):
                return state == states[3]

            def get_successors(self, state):
                return [(SimpleNamespace(value=a), s, c) for a, s, c in edges[state]]

            def transition(self, state, action):
                return next((s, c) for a, s, c in edges[state] if a == action)

        env = Graph()
        h_values = dict(zip(states, (5, 0, 4, 0)))
        result = solver.a_star_search(env, lambda s, world: h_values[s])
        self.assertTrue(result.solved)
        self.assertEqual(result.total_cost, 5)
        self.assertEqual(result.solution_actions, ["SB", "BA", "AG"])
        self.assertEqual(result.nodes_expanded, 4)

    def test_sma_memory_and_regeneration(self):
        env = DeepCoreEnv.from_file_or_builtin("level1")
        for cap in (32, 64, 128):
            result = solver.sma_star_search(env, heuristics.h2_advanced, cap)
            replay(env, result)
            self.assertTrue(result.solved)
            self.assertEqual(result.total_cost, 68)
            self.assertLessEqual(result.max_memory_size, cap)
            self.assertEqual(result.max_memory_size, cap)
        tiny = DeepCoreEnv("#######\n#S.M.E#\n#######")
        self.assertFalse(solver.sma_star_search(tiny, heuristics.h2_advanced, 1).solved)
        self.assertTrue(solver.sma_star_search(tiny, heuristics.h2_advanced, 5).solved)

    def test_limits_and_validation(self):
        env = DeepCoreEnv(SMALL_MAPS["plain"])
        for fn in (solver.depth_first_search, solver.iterative_deepening_search):
            result = fn(env, max_expansions=0)
            self.assertFalse(result.solved)
            self.assertEqual(result.nodes_expanded, 0)
            self.assertEqual(result.status_message, "Expansion limit reached.")
        self.assertFalse(solver.iterative_deepening_search(env, max_depth=3).solved)
        self.assertTrue(solver.iterative_deepening_search(env, max_depth=4).solved)
        self.assertFalse(solver.ida_star_search(env, heuristics.h2_advanced, 0).solved)
        for weight in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                solver.weighted_a_star_search(env, heuristics.h1_basic, weight)
        for cap in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                solver.sma_star_search(env, heuristics.h1_basic, cap)

    def test_already_solved(self):
        env = DeepCoreEnv(SMALL_MAPS["no-miners"])
        env.initial_state = replace(env.initial_state, bot_pos=env.elevator_pos)
        functions = [lambda: solver.breadth_first_search(env),
                     lambda: solver.depth_first_search(env),
                     lambda: solver.iterative_deepening_search(env),
                     lambda: solver.bidirectional_search(env),
                     lambda: solver.uniform_cost_search(env)]
        functions += [lambda fn=fn: fn(env, heuristics.h2_advanced)
                      for fn in (solver.greedy_best_first_search, solver.a_star_search,
                                 solver.weighted_a_star_search, solver.ida_star_search,
                                 solver.sma_star_search)]
        functions.append(lambda: solver.sma_star_search(env, heuristics.h2_advanced, 1))
        for fn in functions:
            result = fn()
            self.assertTrue(result.solved)
            self.assertEqual(result.solution_actions, [])
            self.assertEqual(result.total_cost, 0)

    def test_large_map_analytic_fallback(self):
        # More than 16 dynamic features: the abstract exponential table is skipped.
        env = DeepCoreEnv("#" * 23 + "\n#S" + "M" * 18 + ".E#\n" + "#" * 23)
        state = env.get_initial_state()
        while not env.is_goal(state):
            child, cost = env.transition(state, "MOVE_E")
            for h in (heuristics.h1_basic, heuristics.h2_advanced):
                self.assertLessEqual(h(state, env), cost + h(child, env))
            self.assertLessEqual(heuristics.h1_basic(state, env), heuristics.h2_advanced(state, env))
            state = child
        self.assertIsNone(heuristics._RELAXED_CACHE[env])


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    print(f"Exhaustive properties: {SubmissionChecks.checked_states} states, "
          f"{SubmissionChecks.checked_edges} directed edges.")
    raise SystemExit(not result.wasSuccessful())
