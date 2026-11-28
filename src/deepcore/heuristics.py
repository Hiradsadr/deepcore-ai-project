"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

from typing import Dict, List, Tuple
from collections import deque
import heapq
import math
from itertools import count
from weakref import WeakKeyDictionary
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

# Cache wall-aware shortest path distances per environment instance
_WALL_DIST_CACHE = WeakKeyDictionary()
_RELAXED_CACHE = WeakKeyDictionary()
_MST_CACHE = WeakKeyDictionary()


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """Precomputes all-pairs shortest path steps ignoring dynamic obstacles (only walls # block)."""
    if env in _WALL_DIST_CACHE:
        return _WALL_DIST_CACHE[env]

    walkable = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    ]
    dist_map: Dict[Tuple[Pos, Pos], int] = {}

    for src in walkable:
        q = deque([(src, 0)])
        visited = {src}
        while q:
            curr, d = q.popleft()
            dist_map[(src, curr)] = d
            cr, cc = curr
            for dr, dc in DIRECTION_DELTAS.values():
                nxt = (cr + dr, cc + dc)
                if env.in_bounds(nxt) and nxt not in env.walls and nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))

    _WALL_DIST_CACHE[env] = dist_map
    return dist_map


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    nodes = sorted(set(nodes))
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in sorted(in_mst):
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            return math.inf
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """Consistent terminal-MST attachment plus unavoidable payload transport.

    Keep the robot outside the MST and attach it by its cheapest edge. This
    is at least the all-vertices MST bound, and admits a simple consistency
    proof even when the robot moves or picks up a terminal.
    """
    if env.is_goal(state):
        return 0.0
    return _mst_payload_bound(state, env, _manhattan, "manhattan")


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _mst_payload_bound(state, env, distance, kind):
    targets = state.miners | {env.elevator_pos}
    cache = _MST_CACHE.setdefault(env, {})
    key = (kind, targets)
    if key not in cache:
        cache[key] = _prim_mst_cost(list(targets), distance)
    base = cache[key] + min(distance(state.bot_pos, p) for p in targets)
    # Avoid 0 * infinity on disconnected maps.
    payload = (state.carried * distance(state.bot_pos, env.elevator_pos)
               if state.carried else 0)
    payload += sum(distance(p, env.elevator_pos) for p in state.miners)
    return float(base + env.PAYLOAD_SURCHARGE * payload)


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    def wall(p):
        return not env.in_bounds(p) or p in env.walls
    n_wall = wall((r - 1, c))
    s_wall = wall((r + 1, c))
    e_wall = wall((r, c + 1))
    w_wall = wall((r, c - 1))
    return (n_wall or s_wall) and (e_wall or w_wall)


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """Wall-aware relaxation with incremental opening costs and safe deadlocks.

    On small maps, an exact abstract distance table retains miners, pits and
    rubble, but ignores boulder positions and sludge. A virtual pit opening
    costs PUSH-1=5; remote rubble removal costs 10. A real bridge push maps to
    a normal abstract step plus a 5-cost opening, so it is never counted twice.
    Large custom maps use the consistent analytic bound instead of allocating
    an exponential table. This choice depends only on the fixed map.
    """
    if env.is_goal(state):
        return 0.0
    if _frozen_deadlock(state, env):
        return math.inf
    distances = _get_wall_aware_distances(env)
    wall_distance = lambda a, b: distances.get((a, b), math.inf)
    wall_bound = _mst_payload_bound(state, env, wall_distance, "walls")
    if math.isinf(wall_bound):
        return math.inf
    abstract = _get_relaxed_table(env)
    if abstract is not None and state.carried == env.total_miners - len(state.miners):
        miner_bits, pit_bits, rubble_bits, table = abstract
        key = (state.bot_pos, sum(miner_bits[p] for p in state.miners),
               sum(pit_bits[p] for p in state.chasms),
               sum(rubble_bits[p] for p in state.rubble))
        return float(max(wall_bound, table.get(key, math.inf)))
    return wall_bound + _barrier_bound(state, env)


def _frozen_deadlock(state, env):
    """Only reject when even optimistic connectivity cannot reach a target.

    A corner stone alone is not a deadlock. Movable stones and drillable
    rubble are ignored; pits block only when no usable stone remains.
    """
    frozen = {p for p in state.boulders if _is_corner_deadlock(p, env)}
    no_stones = len(frozen) == len(state.boulders)
    if not frozen and not (no_stones and state.chasms):
        return False
    blocked = env.walls | frozen
    if no_stones:
        blocked = blocked | state.chasms
    seen = {state.bot_pos}
    queue = deque([state.bot_pos])
    targets = state.miners | {env.elevator_pos}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTION_DELTAS.values():
            child = (r + dr, c + dc)
            if env.in_bounds(child) and child not in blocked and child not in seen:
                seen.add(child)
                queue.append(child)
    return not targets <= seen


def _barrier_bound(state, env):
    """Largest minimum opening cost to a required target, with zero walking cost."""
    targets = set(state.miners) | {env.elevator_pos}
    best = {state.bot_pos: 0}
    queue = [(0, state.bot_pos)]
    result = 0
    while queue:
        value, pos = heapq.heappop(queue)
        if value != best[pos]:
            continue
        if pos in targets:
            result = max(result, value)
            targets.remove(pos)
            if not targets:
                return float(result)
        for dr, dc in DIRECTION_DELTAS.values():
            child = (pos[0] + dr, pos[1] + dc)
            if not env.in_bounds(child) or child in env.walls:
                continue
            opening = (env.COST_DRILL if child in state.rubble else
                       env.COST_PUSH_BASE - env.COST_CLEAR_FLOOR if child in state.chasms else 0)
            new_value = value + opening
            if new_value < best.get(child, math.inf):
                best[child] = new_value
                heapq.heappush(queue, (new_value, child))
    return math.inf


def _get_relaxed_table(env):
    if env in _RELAXED_CACHE:
        return _RELAXED_CACHE[env]
    cells = [(r, c) for r in range(env.rows) for c in range(env.cols)
             if (r, c) not in env.walls]
    features = len(env.initial_miners) + len(env.initial_chasms) + len(env.initial_rubble)
    # A preprocessing budget, not a search cutoff. The fallback remains valid.
    if features > 16 or len(cells) * (1 << features) > 100_000:
        _RELAXED_CACHE[env] = None
        return None
    miner_bits = {p: 1 << i for i, p in enumerate(sorted(env.initial_miners))}
    pit_bits = {p: 1 << i for i, p in enumerate(sorted(env.initial_chasms))}
    rubble_bits = {p: 1 << i for i, p in enumerate(sorted(env.initial_rubble))}
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRECTION_DELTAS.values()
                     if env.in_bounds((p[0] + dr, p[1] + dc))
                     and (p[0] + dr, p[1] + dc) not in env.walls] for p in cells}
    best, heap, serial = {}, [], count()

    def relax(key, value):
        if value < best.get(key, math.inf):
            best[key] = value
            heapq.heappush(heap, (value, next(serial), key))

    # Every obstacle configuration at the elevator is an abstract goal.
    for pits in range(1 << len(pit_bits)):
        for rubble in range(1 << len(rubble_bits)):
            relax((env.elevator_pos, 0, pits, rubble), 0)
    while heap:
        value, _, key = heapq.heappop(heap)
        if best[key] != value:
            continue
        pos, miners, pits, rubble = key
        carried = env.total_miners - miners.bit_count()
        pickup_bit = miner_bits.get(pos, 0)
        for previous in neighbors[pos]:
            if (pit_bits.get(previous, 0) & pits
                    or rubble_bits.get(previous, 0) & rubble
                    or miner_bits.get(previous, 0) & miners):
                continue
            relax((previous, miners, pits, rubble),
                  value + env.COST_CLEAR_FLOOR + env.PAYLOAD_SURCHARGE * carried)
            if pickup_bit and not miners & pickup_bit:
                relax((previous, miners | pickup_bit, pits, rubble),
                      value + env.COST_CLEAR_FLOOR + env.PAYLOAD_SURCHARGE * (carried - 1))
        for location, bit in pit_bits.items():
            if not pits & bit and location != pos:
                relax((pos, miners, pits | bit, rubble),
                      value + env.COST_PUSH_BASE - env.COST_CLEAR_FLOOR)
        for location, bit in rubble_bits.items():
            if not rubble & bit and location != pos:
                relax((pos, miners, pits, rubble | bit), value + env.COST_DRILL)
    result = (miner_bits, pit_bits, rubble_bits, best)
    _RELAXED_CACHE[env] = result
    return result
