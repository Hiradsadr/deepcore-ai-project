"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
from itertools import count
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


def _check_budget(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


def _record_memory(metrics, frontier_size, memory_size):
    metrics.max_frontier_size = max(metrics.max_frontier_size, frontier_size)
    metrics.max_memory_size = max(metrics.max_memory_size, memory_size)


def _record_expansion(metrics, state):
    metrics.nodes_expanded += 1
    metrics.explored_order.append(state.bot_pos)


def _failed(metrics, message="No solution found."):
    metrics.status_message = message
    return metrics


def _budget_reached(metrics, limit):
    if limit is not None and metrics.nodes_expanded >= limit:
        metrics.status_message = "Expansion limit reached."
        return True
    return False


def _solved(metrics, parent, state, cost, env=None):
    metrics.solved = True
    metrics.solution_actions = _reconstruct_path(parent, state)
    metrics.total_cost = cost
    if env is not None:
        current, metrics.total_cost = env.get_initial_state(), 0
        for action in metrics.solution_actions:
            current, step_cost = env.transition(current, action)
            metrics.total_cost += step_cost
    metrics.status_message = "Solved."
    return metrics


def _reachable_state_invariants(env, state):
    """Reject impossible reverse states without changing the provided environment."""
    return (
        state.carried == env.total_miners - len(state.miners)
        and len(state.boulders) - len(state.chasms)
        == len(env.initial_boulders) - len(env.initial_chasms)
        and env.in_bounds(state.bot_pos)
        and state.bot_pos not in env.walls
        and state.bot_pos not in state.miners | state.boulders | state.chasms | state.rubble
        and state.miners <= env.initial_miners
        and state.chasms <= env.initial_chasms
        and state.rubble <= env.initial_rubble
        and not state.boulders.intersection(state.chasms | state.rubble | state.miners | env.walls)
        and env.elevator_pos not in state.boulders
    )


def _best_first(env, heuristic, label, mode, weight=1.0):
    metrics = SearchMetrics(algorithm_name=label)
    start = env.get_initial_state()
    parent, best_g = {start: (None, None)}, {start: 0}
    serial = count()
    start_h = float(heuristic(start, env))
    if math.isnan(start_h) or start_h < 0:
        raise ValueError("heuristic values must be non-negative and not NaN")
    if math.isinf(start_h) and not env.is_goal(start):
        return _failed(metrics, "No solution: heuristic proves a deadlock.")
    heap = [(weight * start_h, next(serial), 0, start)]
    open_nodes = {start: 0}
    while heap:
        _record_memory(metrics, len(heap), len(best_g) + len(heap) - len(open_nodes))
        _, _, g, state = heapq.heappop(heap)
        if g != best_g[state] or open_nodes.get(state) != g:
            continue
        del open_nodes[state]
        if env.is_goal(state):
            return _solved(metrics, parent, state, g, env)
        _record_expansion(metrics, state)
        for action, child, cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_g = g + cost
            if new_g >= best_g.get(child, math.inf):
                continue
            h = float(heuristic(child, env))
            if math.isnan(h) or h < 0:
                raise ValueError("heuristic values must be non-negative and not NaN")
            if math.isinf(h):
                continue
            best_g[child] = new_g
            parent[child] = (state, action.value)
            open_nodes[child] = new_g
            priority = h if mode == "greedy" else new_g + weight * h
            heapq.heappush(heap, (priority, next(serial), new_g, child))
        _record_memory(metrics, len(heap), len(best_g) + len(heap) - len(open_nodes))
        # Discard obsolete heap records without changing priority/tie order.
        if len(heap) > 2 * len(open_nodes) + 64:
            heap[:] = [entry for entry in heap if open_nodes.get(entry[3]) == entry[2]]
            heapq.heapify(heap)
    return _failed(metrics)


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS)
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    """LIFO graph search. An explicit budget is optional, never implicit."""
    _check_budget(max_expansions, "max_expansions")
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    stack = [start]
    parent = {start: (None, None)}
    costs = {start: 0}
    while stack:
        _record_memory(metrics, len(stack), len(parent))
        state = stack.pop()
        if env.is_goal(state):
            return _solved(metrics, parent, state, costs[state])
        if _budget_reached(metrics, max_expansions):
            return metrics
        _record_expansion(metrics, state)
        successors = env.get_successors(state)
        metrics.nodes_generated += len(successors)
        # Reverse insertion preserves the environment's N,S,E,W visit order.
        for action, child, cost in reversed(successors):
            if child not in parent:
                parent[child] = (state, action.value)
                costs[child] = costs[state] + cost
                stack.append(child)
        _record_memory(metrics, len(stack), len(parent))
    return _failed(metrics)


def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """Depth-limited DFS passes with path cycles and depth dominance checks.

    The per-pass best-depth table is reset at every iteration. It trades the
    textbook tree version's linear memory for practical graph-search pruning.
    Explicit frames avoid Python's recursion-depth limit.
    """
    _check_budget(max_depth, "max_depth")
    _check_budget(max_expansions, "max_expansions")
    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    limit = 0
    while max_depth is None or limit <= max_depth:
        parent = {start: (None, None)}
        best_depth = {start: 0}
        # A frame is [state, depth, accumulated cost, successor iterator].
        frames = [[start, 0, 0, None]]
        cutoff = False
        while frames:
            state, depth, cost, children = frames[-1]
            _record_memory(metrics, len(frames), len(best_depth))
            if children is None:
                if env.is_goal(state):
                    return _solved(metrics, parent, state, cost)
                if depth == limit:
                    cutoff = True
                    frames.pop()
                    del parent[state]
                    continue
                if _budget_reached(metrics, max_expansions):
                    return metrics
                _record_expansion(metrics, state)
                successors = env.get_successors(state)
                metrics.nodes_generated += len(successors)
                frames[-1][3] = iter(successors)
            try:
                action, child, step_cost = next(frames[-1][3])
            except StopIteration:
                frames.pop()
                del parent[state]
                continue
            if child in parent or best_depth.get(child, math.inf) <= depth + 1:
                continue
            best_depth[child] = depth + 1
            parent[child] = (state, action.value)
            frames.append([child, depth + 1, cost + step_cost, None])
        if not cutoff:
            return _failed(metrics)
        limit += 1
    return _failed(metrics, "Depth limit reached.")


# =============================================================================
# 3. Bidirectional Search — [STUDENT TODO]
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """Layered two-ended BFS, with a forward certificate for arbitrary goals.

    The supplied environment lists only a few canonical goals, not every goal.
    A meeting therefore gives an upper bound, not a safe stopping condition.
    Forward layers certify that no shorter *noncanonical* goal was missed.
    """
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    forward = deque([start])
    backward = deque(dict.fromkeys(env.get_goal_states()))
    f_parent = {start: (None, None)}
    b_parent = {goal: (None, None) for goal in backward}
    f_depth, b_depth = {start: 0}, {goal: 0 for goal in backward}
    f_cost, b_cost = {start: 0}, {goal: 0 for goal in backward}
    best_meeting, best_length = None, math.inf

    def consider(state):
        nonlocal best_meeting, best_length
        if state in f_depth and state in b_depth:
            length = f_depth[state] + b_depth[state]
            if length < best_length:
                best_meeting, best_length = state, length

    def joined():
        state = best_meeting
        actions = _reconstruct_path(f_parent, state)
        while b_parent[state][0] is not None:
            next_state, action = b_parent[state]
            actions.append(action)  # Already oriented from predecessor to goal.
            state = next_state
        metrics.solved = True
        metrics.solution_actions = actions
        metrics.total_cost = f_cost[best_meeting] + b_cost[best_meeting]
        metrics.status_message = "Solved; minimum action count certified."
        return metrics

    consider(start)
    while forward:
        _record_memory(metrics, len(forward) + len(backward), len(f_parent) + len(b_parent))
        if env.is_goal(forward[0]):
            state = forward[0]
            return _solved(metrics, f_parent, state, f_cost[state])
        # All forward goals through this depth have already been tested on discovery.
        if best_meeting is not None and f_depth[forward[0]] + 1 >= best_length:
            return joined()
        # Once a meeting is known, only the forward certificate is still needed.
        expand_forward = (best_meeting is not None or not backward
                          or f_depth[forward[0]] <= b_depth[backward[0]])
        queue = forward if expand_forward else backward
        depths = f_depth if expand_forward else b_depth
        layer = depths[queue[0]]
        while queue and depths[queue[0]] == layer:
            state = queue.popleft()
            if not expand_forward and not _reachable_state_invariants(env, state):
                continue
            _record_expansion(metrics, state)
            transitions = (env.get_successors(state) if expand_forward
                           else env.get_predecessors(state))
            metrics.nodes_generated += len(transitions)
            for action, child, cost in transitions:
                if child in depths:
                    continue
                if not expand_forward and not _reachable_state_invariants(env, child):
                    continue
                depths[child] = layer + 1
                queue.append(child)
                if expand_forward:
                    f_parent[child] = (state, action.value)
                    f_cost[child] = f_cost[state] + cost
                    if env.is_goal(child):
                        _record_memory(metrics, len(forward) + len(backward), len(f_parent) + len(b_parent))
                        return _solved(metrics, f_parent, child, f_cost[child])
                else:
                    b_parent[child] = (state, action.value)
                    b_cost[child] = b_cost[state] + cost
                consider(child)
            _record_memory(metrics, len(forward) + len(backward), len(f_parent) + len(b_parent))
    return joined() if best_meeting is not None else _failed(metrics)


# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """Dijkstra graph search, with the goal test on priority-queue removal."""
    return _best_first(env, lambda state, world: 0.0, "UCS", "cost")


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """Best-first search whose priority is only h; g is used for bookkeeping."""
    return _best_first(env, heuristic, "GBFS", "greedy")


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """A* with best-g reopening, stale-entry rejection and FIFO tie breaking."""
    return _best_first(env, heuristic, "A*", "cost")


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """Weighted A*: weight=1 is A*; larger weights trade cost for search effort."""
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("weight must be finite and at least 1")
    return _best_first(env, heuristic, "Weighted A*", "cost", weight)


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    raise NotImplementedError


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError
