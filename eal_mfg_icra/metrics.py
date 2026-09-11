"""Auditable trajectory metrics for local flocking experiments."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from .core import NeighborGraph, PopulationState, Rollout, _norm_sq


def neighbor_loss_rate(neighbor_history: Sequence[NeighborGraph]) -> float:
    """Fraction of agent transitions that lose at least one neighbour.

    This is the metric used by the first flocking paper.  The denominator is
    the number of agent transitions, not the number of agents that happened to
    have neighbours at the start; an isolated agent is therefore a valid,
    non-losing transition.
    """

    if len(neighbor_history) < 2:
        return 0.0
    n_agents = len(neighbor_history[0])
    if n_agents == 0:
        return 0.0
    transitions = 0
    losses = 0
    for before, after in zip(neighbor_history, neighbor_history[1:]):
        if len(before) != n_agents or len(after) != n_agents:
            raise ValueError("neighbor graph sizes must remain constant")
        for i in range(n_agents):
            transitions += 1
            # The first paper defines cost by a decrease in neighbour count,
            # not by identity replacement.  Keep that definition exact.
            if len(after[i]) < len(before[i]):
                losses += 1
    return losses / transitions if transitions else 0.0


def contact_break_rate(neighbor_history: Sequence[NeighborGraph]) -> float:
    """Fraction of transitions where at least one previous contact disappears.

    This stricter diagnostic is intentionally separate from
    :func:`neighbor_loss_rate`; a replacement can preserve cardinality while
    still breaking an individual contact.
    """

    if len(neighbor_history) < 2:
        return 0.0
    n_agents = len(neighbor_history[0])
    transitions = 0
    breaks = 0
    for before, after in zip(neighbor_history, neighbor_history[1:]):
        if len(before) != n_agents or len(after) != n_agents:
            raise ValueError("neighbor graph sizes must remain constant")
        for i in range(n_agents):
            transitions += 1
            if set(before[i]).difference(after[i]):
                breaks += 1
    return breaks / transitions if transitions else 0.0


def polar_order(states: Sequence[PopulationState]) -> float:
    """Mean Vicsek polar order in ``[0, 1]`` over a state trajectory.

    Each nonzero velocity contributes its unit heading.  A zero-velocity agent
    contributes the zero vector rather than an arbitrary heading, making the
    metric explicit under braking or clipped dynamics.
    """

    if not states:
        return 0.0
    values = []
    for state in states:
        if state.n_agents == 0:
            values.append(0.0)
            continue
        heading_sum = [0.0] * state.dim
        for velocity in state.velocities:
            speed = sqrt(_norm_sq(velocity))
            if speed <= 1e-12:
                continue
            for k in range(state.dim):
                heading_sum[k] += velocity[k] / speed
        values.append(sqrt(sum(value * value for value in heading_sum)) / state.n_agents)
    return sum(values) / len(values)


def connected_components(
    graph: NeighborGraph, symmetrize: bool = True
) -> tuple[tuple[int, ...], ...]:
    """Return connected components of a possibly directed local graph."""

    n_agents = len(graph)
    adjacency: list[set[int]] = [set() for _ in range(n_agents)]
    for i, local in enumerate(graph):
        for j in local:
            if j < 0 or j >= n_agents or j == i:
                raise ValueError("neighbor index out of range or self-loop")
            adjacency[i].add(j)
            if symmetrize:
                adjacency[j].add(i)
    unseen = set(range(n_agents))
    components = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: item[0]))


def fragmentation_score(graph: NeighborGraph) -> dict[str, float]:
    """Summarize connectedness of one local graph.

    ``fragmentation`` is one minus the largest-component fraction.  It is 0
    for a connected population and approaches 1 when most agents are isolated.
    ``components`` is returned as a float for easy JSON/CSV serialization.
    """

    n_agents = len(graph)
    if n_agents == 0:
        return {"components": 0.0, "largest_component_fraction": 0.0, "fragmentation": 0.0}
    components = connected_components(graph)
    largest = max(len(component) for component in components)
    largest_fraction = largest / n_agents
    return {
        "components": float(len(components)),
        "largest_component_fraction": largest_fraction,
        "fragmentation": 1.0 - largest_fraction,
    }


def velocity_dispersion(states: Sequence[PopulationState]) -> float:
    """Mean per-state velocity variance (diagnostic, not a training reward)."""

    if not states:
        return 0.0
    values = []
    for state in states:
        mean = [
            sum(velocity[k] for velocity in state.velocities) / state.n_agents
            for k in range(state.dim)
        ]
        values.append(
            sum(_norm_sq(tuple(velocity[k] - mean[k] for k in range(state.dim))) for velocity in state.velocities)
            / state.n_agents
        )
    return sum(values) / len(values)


def mean_interaction_degree(neighbor_history: Sequence[NeighborGraph]) -> float:
    """Mean realized directed out-degree over agents and graph snapshots."""

    total = 0
    count = 0
    for graph in neighbor_history:
        total += sum(len(local) for local in graph)
        count += len(graph)
    return total / count if count else 0.0


def mean_interaction_radius(radius_history: Sequence[Sequence[float]]) -> float:
    """Mean realized interaction radius over agents and graph snapshots."""

    total = 0.0
    count = 0
    for radii in radius_history:
        total += sum(float(radius) for radius in radii)
        count += len(radii)
    return total / count if count else 0.0


def strongly_connected_components(graph: NeighborGraph) -> tuple[tuple[int, ...], ...]:
    """Return strongly connected components of a directed local graph.

    Uses iterative Tarjan with explicit stacks (safe for the small populations
    used in this project).  ``graph[i]`` lists out-neighbours of ``i``.
    """

    n_agents = len(graph)
    for i, local in enumerate(graph):
        for j in local:
            if j < 0 or j >= n_agents or j == i:
                raise ValueError("neighbor index out of range or self-loop")
    indices = [-1] * n_agents
    lowlink = [0] * n_agents
    on_stack = [False] * n_agents
    component_stack: list[int] = []
    components: list[tuple[int, ...]] = []
    next_index = 0
    for root in range(n_agents):
        if indices[root] != -1:
            continue
        call_stack: list[tuple[int, int]] = [(root, 0)]
        while call_stack:
            node, neighbor_pos = call_stack[-1]
            if neighbor_pos == 0:
                indices[node] = next_index
                lowlink[node] = next_index
                next_index += 1
                component_stack.append(node)
                on_stack[node] = True
            advanced = False
            local = graph[node]
            while neighbor_pos < len(local):
                child = local[neighbor_pos]
                call_stack[-1] = (node, neighbor_pos + 1)
                if indices[child] == -1:
                    call_stack.append((child, 0))
                    advanced = True
                    break
                if on_stack[child]:
                    lowlink[node] = min(lowlink[node], indices[child])
                neighbor_pos += 1
            if advanced:
                continue
            if not advanced and neighbor_pos == len(local):
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == indices[node]:
                    component: list[int] = []
                    while True:
                        member = component_stack.pop()
                        on_stack[member] = False
                        component.append(member)
                        if member == node:
                            break
                    components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: item[0]))


def scc_summary(graph: NeighborGraph) -> dict[str, float]:
    """Summarize the directed connectivity structure of one graph snapshot.

    ``largest_scc_fraction`` is the fraction of agents in the largest strongly
    connected component; it is 1.0 when every agent can reach every other via
    directed paths.
    """

    n_agents = len(graph)
    if n_agents == 0:
        return {"scc_components": 0.0, "largest_scc_fraction": 0.0}
    components = strongly_connected_components(graph)
    largest = max(len(component) for component in components)
    return {
        "scc_components": float(len(components)),
        "largest_scc_fraction": largest / n_agents,
    }


def directed_degree_stats(graph: NeighborGraph) -> dict[str, float]:
    """Mean/min in- and out-degree of the realized directed graph."""

    n_agents = len(graph)
    if n_agents == 0:
        return {
            "mean_out_degree": 0.0,
            "min_out_degree": 0.0,
            "mean_in_degree": 0.0,
            "min_in_degree": 0.0,
        }
    out_degrees = [len(local) for local in graph]
    in_degrees = [0] * n_agents
    for local in graph:
        for j in local:
            in_degrees[j] += 1
    return {
        "mean_out_degree": sum(out_degrees) / n_agents,
        "min_out_degree": float(min(out_degrees)),
        "mean_in_degree": sum(in_degrees) / n_agents,
        "min_in_degree": float(min(in_degrees)),
    }


def rooted_fraction(graph: NeighborGraph) -> float:
    """Fraction of agents that can reach every other agent via directed paths.

    A value of 1.0 means the directed graph has a spanning arborescence from
    every possible root; this is the directed analogue of the largest-component
    diagnostic and is reported as a reachability diagnostic only.
    """

    n_agents = len(graph)
    if n_agents == 0:
        return 0.0
    if n_agents == 1:
        return 1.0
    rooted = 0
    for root in range(n_agents):
        seen = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for child in graph[node]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        if len(seen) == n_agents:
            rooted += 1
    return rooted / n_agents


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Eigenvalues of a symmetric matrix via Jacobi rotations (stdlib only).

    Sufficient for the small graph Laplacians (N <= 64) used by this project.
    """

    from math import sqrt

    n_agents = len(matrix)
    if n_agents == 0:
        return []
    if n_agents == 1:
        return [float(matrix[0][0])]
    values = [list(map(float, row)) for row in matrix]
    for _ in range(100):
        largest = 0.0
        p = q = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                magnitude = abs(values[i][j])
                if magnitude > largest:
                    largest, p, q = magnitude, i, j
        if largest < 1e-12:
            break
        theta = (values[q][q] - values[p][p]) / (2.0 * values[p][q])
        if theta >= 0.0:
            t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0))
        else:
            t = -1.0 / (abs(theta) + sqrt(theta * theta + 1.0))
        c = 1.0 / sqrt(t * t + 1.0)
        s = t * c
        tau = s / (1.0 + c)
        app, aqq, apq = values[p][p], values[q][q], values[p][q]
        for k in range(n_agents):
            if k == p or k == q:
                continue
            akp, akq = values[k][p], values[k][q]
            values[k][p] = c * akp - s * akq
            values[k][q] = s * akp + c * akq
            values[p][k] = values[k][p]
            values[q][k] = values[k][q]
        values[p][p] = app - t * apq
        values[q][q] = aqq + t * apq
        values[p][q] = 0.0
        values[q][p] = 0.0
    return sorted(values[i][i] for i in range(n_agents))


def symmetrized_algebraic_connectivity(graph: NeighborGraph) -> float:
    """Second smallest Laplacian eigenvalue of the symmetrized support.

    The symmetrized graph is ``A_sym = A OR A^T``; lambda_2 of its Laplacian
    is an *undirected* connectivity diagnostic.  It is deliberately NOT the
    algebraic connectivity of the directed graph, which is not well defined in
    the same sense; treating it as such would be a category error.
    """

    n_agents = len(graph)
    if n_agents < 2:
        return 0.0
    adjacency = [[0.0] * n_agents for _ in range(n_agents)]
    for i, local in enumerate(graph):
        for j in local:
            adjacency[i][j] = 1.0
            adjacency[j][i] = 1.0
    laplacian = [[0.0] * n_agents for _ in range(n_agents)]
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                laplacian[i][j] = sum(adjacency[i])
            else:
                laplacian[i][j] = -adjacency[i][j]
    eigenvalues = _symmetric_eigenvalues(laplacian)
    return eigenvalues[1] if len(eigenvalues) > 1 else 0.0


def communication_cost(graph: NeighborGraph) -> dict[str, float]:
    """Interaction-workload proxy for one directed graph snapshot.

    ``edge_count`` / ``message_count`` count directed edges; if every agent
    transmitted one message per neighbour per step, message_count would be the
    exact message volume.  This is an interaction-workload proxy, not a
    byte-level bandwidth estimate.
    """

    n_agents = len(graph)
    edge_count = sum(len(local) for local in graph)
    maximum = n_agents * (n_agents - 1) if n_agents > 0 else 0
    return {
        "edge_count": float(edge_count),
        "normalized_comm_cost": edge_count / maximum if maximum else 0.0,
        "message_count": float(edge_count),
    }


def edge_persistence(neighbor_history: Sequence[NeighborGraph]) -> float:
    """Mean fraction of directed edges retained between consecutive snapshots.

    Complement of identity-based churn: 1.0 means every previous edge survives
    the transition (or there was nothing to lose); lower values mean the
    topology is churning.
    """

    if len(neighbor_history) < 2:
        return 0.0
    n_agents = len(neighbor_history[0])
    if n_agents == 0:
        return 0.0
    total = 0.0
    count = 0
    for before, after in zip(neighbor_history, neighbor_history[1:]):
        if len(before) != n_agents or len(after) != n_agents:
            raise ValueError("neighbor graph sizes must remain constant")
        for i in range(n_agents):
            previous = set(before[i])
            if not previous:
                continue
            total += len(previous.intersection(after[i])) / len(previous)
            count += 1
    return total / count if count else 0.0


def compute_connectivity_metrics(rollout: Rollout) -> dict[str, float]:
    """Directed-graph connectivity diagnostics plus interaction-workload proxies.

    Every mean-in-time quantity averages over all graph snapshots including the
    initial one, mirroring ``mean_fragmentation``.  Directed quantities use the
    realized directed graph; the algebraic connectivity is an undirected
    diagnostic of the symmetrized support and is explicitly named as such.
    """

    graphs = tuple(rollout.neighbor_history)
    if not graphs:
        return {
            "final_largest_scc_fraction": 0.0,
            "final_n_scc": 0.0,
            "mean_largest_scc_fraction": 0.0,
            "mean_n_scc": 0.0,
            "final_mean_out_degree": 0.0,
            "final_mean_in_degree": 0.0,
            "final_min_out_degree": 0.0,
            "final_min_in_degree": 0.0,
            "mean_min_out_degree": 0.0,
            "mean_min_in_degree": 0.0,
            "final_sym_lambda2": 0.0,
            "mean_sym_lambda2": 0.0,
            "final_edge_count": 0.0,
            "mean_edge_count": 0.0,
            "final_normalized_comm_cost": 0.0,
            "mean_normalized_comm_cost": 0.0,
            "final_rooted_fraction": 0.0,
            "mean_rooted_fraction": 0.0,
            "edge_persistence": 0.0,
            "mean_radius_sq": 0.0,
            "mean_radius_footprint": 0.0,
        }
    final = graphs[-1]
    final_scc = scc_summary(final)
    final_degrees = directed_degree_stats(final)
    final_cost = communication_cost(final)

    def mean_of(fn):
        return sum(fn(snapshot) for snapshot in graphs) / len(graphs)

    scc_values = [scc_summary(snapshot) for snapshot in graphs]
    degree_values = [directed_degree_stats(snapshot) for snapshot in graphs]
    cost_values = [communication_cost(snapshot) for snapshot in graphs]
    radius_sq_total = 0.0
    radius_sq_count = 0
    footprint_total = 0.0
    for radii in rollout.radius_history:
        footprint_total += sum(float(r) * float(r) for r in radii)
        radius_sq_total += sum(float(r) * float(r) for r in radii)
        radius_sq_count += len(radii)
    return {
        "final_largest_scc_fraction": final_scc["largest_scc_fraction"],
        "final_n_scc": final_scc["scc_components"],
        "mean_largest_scc_fraction": sum(v["largest_scc_fraction"] for v in scc_values) / len(scc_values),
        "mean_n_scc": sum(v["scc_components"] for v in scc_values) / len(scc_values),
        "final_mean_out_degree": final_degrees["mean_out_degree"],
        "final_mean_in_degree": final_degrees["mean_in_degree"],
        "final_min_out_degree": final_degrees["min_out_degree"],
        "final_min_in_degree": final_degrees["min_in_degree"],
        "mean_min_out_degree": sum(v["min_out_degree"] for v in degree_values) / len(degree_values),
        "mean_min_in_degree": sum(v["min_in_degree"] for v in degree_values) / len(degree_values),
        "final_sym_lambda2": symmetrized_algebraic_connectivity(final),
        "mean_sym_lambda2": mean_of(symmetrized_algebraic_connectivity),
        "final_edge_count": final_cost["edge_count"],
        "mean_edge_count": sum(v["edge_count"] for v in cost_values) / len(cost_values),
        "final_normalized_comm_cost": final_cost["normalized_comm_cost"],
        "mean_normalized_comm_cost": sum(v["normalized_comm_cost"] for v in cost_values) / len(cost_values),
        "final_rooted_fraction": rooted_fraction(final),
        "mean_rooted_fraction": mean_of(rooted_fraction),
        "edge_persistence": edge_persistence(graphs),
        "mean_radius_sq": radius_sq_total / radius_sq_count if radius_sq_count else 0.0,
        "mean_radius_footprint": footprint_total / len(graphs) if graphs else 0.0,
    }


def compute_metrics(rollout: Rollout) -> dict[str, float]:
    """Compute scalar metrics from a :class:`~eal_mfg_icra.core.Rollout`.

    ``fragmentation`` and ``largest_component_fraction`` are retained as
    compatibility aliases for the final-graph values.  New reports should use
    the explicit ``final_*`` and ``mean_*`` names: the former describes the
    terminal graph while the latter averages every graph snapshot in the
    trajectory, including the initial snapshot.
    """

    if not rollout.states:
        return {
            "neighbor_loss_rate": 0.0,
            "contact_break_rate": 0.0,
            "polar_order": 0.0,
            "components": 0.0,
            "largest_component_fraction": 0.0,
            "fragmentation": 0.0,
            "final_fragmentation": 0.0,
            "mean_fragmentation": 0.0,
            "mean_largest_component_fraction": 0.0,
            "mean_degree": 0.0,
            "mean_interaction_out_degree": 0.0,
            "mean_radius": 0.0,
            "velocity_dispersion": 0.0,
            "agent_collision_rate": 0.0,
            "obstacle_collision_rate": 0.0,
            "obstacle_proximity_rate": 0.0,
            "minimum_agent_distance": 0.0,
            "minimum_obstacle_clearance": 0.0,
        }
    graph_history = tuple(rollout.neighbor_history)
    graph = graph_history[-1] if graph_history else tuple()
    final = fragmentation_score(graph)
    trajectory = tuple(fragmentation_score(snapshot) for snapshot in graph_history)
    mean_fragmentation = (
        sum(item["fragmentation"] for item in trajectory) / len(trajectory)
        if trajectory
        else 0.0
    )
    mean_largest = (
        sum(item["largest_component_fraction"] for item in trajectory) / len(trajectory)
        if trajectory
        else 0.0
    )
    safety = tuple(rollout.safety_history)
    agent_collision_rate = (
        sum(item["agent_collision_rate"] for item in safety) / len(safety)
        if safety
        else 0.0
    )
    obstacle_collision_rate = (
        sum(item["obstacle_collision_rate"] for item in safety) / len(safety)
        if safety
        else 0.0
    )
    obstacle_proximity_rate = (
        sum(item["obstacle_proximity_rate"] for item in safety) / len(safety)
        if safety
        else 0.0
    )
    minimum_agent_distance = (
        min(item["minimum_agent_distance"] for item in safety)
        if safety
        else 0.0
    )
    minimum_obstacle_clearance = (
        min(item["minimum_obstacle_clearance"] for item in safety)
        if safety
        else 0.0
    )
    realized_degree = mean_interaction_degree(graph_history)
    realized_radius = mean_interaction_radius(rollout.radius_history)
    return {
        "neighbor_loss_rate": neighbor_loss_rate(rollout.neighbor_history),
        "contact_break_rate": contact_break_rate(rollout.neighbor_history),
        "polar_order": polar_order(rollout.states),
        "components": final["components"],
        "largest_component_fraction": final["largest_component_fraction"],
        "fragmentation": final["fragmentation"],
        "final_fragmentation": final["fragmentation"],
        "mean_fragmentation": mean_fragmentation,
        "mean_largest_component_fraction": mean_largest,
        # ``mean_degree`` is the requested compatibility name. It denotes the
        # realized directed interaction graph, not the fixed-r_b probe degree
        # that drives the current radius implementation.
        "mean_degree": realized_degree,
        "mean_interaction_out_degree": realized_degree,
        "mean_radius": realized_radius,
        "velocity_dispersion": velocity_dispersion(rollout.states),
        "agent_collision_rate": agent_collision_rate,
        "obstacle_collision_rate": obstacle_collision_rate,
        "obstacle_proximity_rate": obstacle_proximity_rate,
        "minimum_agent_distance": minimum_agent_distance,
        "minimum_obstacle_clearance": minimum_obstacle_clearance,
    }


__all__ = [
    "compute_metrics",
    "compute_connectivity_metrics",
    "connected_components",
    "fragmentation_score",
    "neighbor_loss_rate",
    "contact_break_rate",
    "polar_order",
    "velocity_dispersion",
    "mean_interaction_degree",
    "mean_interaction_radius",
    "strongly_connected_components",
    "scc_summary",
    "directed_degree_stats",
    "rooted_fraction",
    "symmetrized_algebraic_connectivity",
    "communication_cost",
    "edge_persistence",
]
