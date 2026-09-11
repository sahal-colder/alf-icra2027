"""Core mechanics for the small EAL-MFG prototype.

The implementation intentionally uses only the Python standard library.  It
keeps the state as tuples/lists so that the semantics remain inspectable in a
debugger and so NumPy is not a mandatory dependency.  The simulator is not a
complete RL trainer: it is the deterministic environment/actor scaffold that
lets us test local mean-field assumptions before adding a learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, floor, isfinite, log1p, pi, sqrt, tanh
from numbers import Integral
import random
from typing import Iterable, Optional, Protocol, Sequence


Number = float
Vector = tuple[Number, ...]
NeighborGraph = tuple[tuple[int, ...], ...]


class PopulationActor(Protocol):
    """Structural actor interface accepted by the rollout harness."""

    def act_population(
        self,
        state: "PopulationState",
        neighborhood: "DensityAdaptiveNeighborhood",
        graph: Optional[NeighborGraph] = None,
        radii: Optional[Sequence[float]] = None,
    ) -> tuple[Vector, ...]: ...


def _vector(values: Iterable[Number], dim: Optional[int] = None) -> Vector:
    result = tuple(float(value) for value in values)
    if dim is not None and len(result) != dim:
        raise ValueError(f"expected dimension {dim}, got {len(result)}")
    if not all(isfinite(value) for value in result):
        raise ValueError("state/action vectors must contain finite values")
    return result


def _zeros(dim: int) -> Vector:
    return (0.0,) * dim


def _norm_sq(vector: Sequence[Number]) -> float:
    return sum(value * value for value in vector)


def _dot(left: Sequence[Number], right: Sequence[Number]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clip_vector(vector: Sequence[Number], lower: float, upper: float) -> Vector:
    return tuple(_clip(float(value), lower, upper) for value in vector)


def _add_scaled(left: Sequence[Number], right: Sequence[Number], scale: float) -> Vector:
    return tuple(a + scale * b for a, b in zip(left, right))


def _sub(left: Sequence[Number], right: Sequence[Number]) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def _mean_vectors(vectors: Sequence[Sequence[Number]], dim: int) -> Vector:
    if not vectors:
        return _zeros(dim)
    return tuple(sum(vector[k] for vector in vectors) / len(vectors) for k in range(dim))


def _toroidal_delta(left: Sequence[Number], right: Sequence[Number], half_width: float) -> Vector:
    """Shortest displacement ``left - right`` on a ``[-L, L]`` torus."""

    period = 2.0 * half_width
    result = []
    for a, b in zip(left, right):
        delta = float(a) - float(b)
        # Using floor instead of round gives deterministic behaviour at the
        # half-period boundary (and avoids Python's bankers-rounding surprise).
        delta -= period * floor(delta / period + 0.5)
        result.append(delta)
    return tuple(result)


@dataclass(frozen=True)
class PopulationState:
    """Position/velocity state for a homogeneous population."""

    positions: tuple[Vector, ...] | Sequence[Sequence[Number]]
    velocities: tuple[Vector, ...] | Sequence[Sequence[Number]]

    def __post_init__(self) -> None:
        positions = tuple(_vector(position) for position in self.positions)
        velocities = tuple(_vector(velocity) for velocity in self.velocities)
        if len(positions) == 0:
            raise ValueError("population must contain at least one agent")
        if len(positions) != len(velocities):
            raise ValueError("positions and velocities must have equal length")
        dim = len(positions[0])
        if dim == 0:
            raise ValueError("state dimension must be positive")
        if any(len(position) != dim for position in positions):
            raise ValueError("all positions must share one dimension")
        if any(len(velocity) != dim for velocity in velocities):
            raise ValueError("all velocities must share the position dimension")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)

    @property
    def n_agents(self) -> int:
        return len(self.positions)

    @property
    def dim(self) -> int:
        return len(self.positions[0])

    def reordered(self, order: Sequence[int]) -> "PopulationState":
        """Return a state with agents permuted by ``order`` (for tests)."""

        if sorted(order) != list(range(self.n_agents)):
            raise ValueError("order must be a permutation of agent indices")
        return PopulationState(
            tuple(self.positions[index] for index in order),
            tuple(self.velocities[index] for index in order),
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Physical and local-kernel parameters.

    ``space_half_width`` defines a periodic box ``[-L, L]^dim``.  The default
    dynamics follow the paper's ordering: position uses ``v_t`` and velocity
    uses ``v_t + u_t dt + epsilon``.  The local flocking term is the paper's
    inverse-distance kernel, truncated to an adaptive radius.  If an agent has
    no local neighbours its flocking term is exactly zero; there is no hidden
    global or random-K fallback.
    """

    dim: int = 2
    space_half_width: float = 100.0
    dt: float = 1.0
    velocity_limit: float = 1.0
    acceleration_limit: float = 1.0
    noise_std: float = 0.0
    beta: float = 1.0
    base_radius: float = 10.0
    target_degree: float = 8.0
    min_radius: float = 1.0
    max_radius: float = 40.0
    radius_exponent: float = 0.5
    # Fraction of the previous radius retained at each step.  Zero preserves
    # the memoryless adaptive rule; values near one add temporal hysteresis.
    radius_smoothing: float = 0.0
    # Extra distance allowed for an already active contact.  New contacts still
    # use the current radius; this margin only prevents boundary chatter.
    contact_retention_margin: float = 0.0
    control_weight: float = 1.0
    velocity_bonus_weight: float = 0.0
    position_target: Optional[float] = None
    position_weight: float = 0.0
    # Optional local transition shaping shared by ALF and the diagnostic
    # contact-residual policy.  Both terms are zero by default so the
    # paper-parity reward is unchanged for existing baselines.
    contact_reward_weight: float = 0.0
    degree_reward_weight: float = 0.0
    # Optional soft safety constraints. Obstacles are static discs on the
    # periodic domain. All terms are disabled by default so historical open
    # flocking runs retain exactly the same transition and reward semantics.
    agent_collision_distance: float = 0.0
    collision_reward_weight: float = 0.0
    obstacle_centers: tuple[Vector, ...] | Sequence[Sequence[Number]] = ()
    obstacle_radius: float = 1.0
    obstacle_safety_margin: float = 0.0
    obstacle_reward_weight: float = 0.0
    obstacle_sensor_range: float = 4.0
    # ``population`` preserves the 1/N factor in the paper's empirical sum;
    # ``local_mean`` is useful when comparing density-matched local systems.
    flocking_normalization: str = "population"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        for name in (
            "space_half_width",
            "dt",
            "velocity_limit",
            "acceleration_limit",
            "base_radius",
            "target_degree",
            "min_radius",
            "max_radius",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("noise_std", "beta", "radius_exponent"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(float(self.radius_smoothing)) or not 0.0 <= self.radius_smoothing <= 1.0:
            raise ValueError("radius_smoothing must be finite and in [0, 1]")
        if not isfinite(float(self.contact_retention_margin)) or self.contact_retention_margin < 0.0:
            raise ValueError("contact_retention_margin must be finite and non-negative")
        for name in (
            "control_weight",
            "position_weight",
            "contact_reward_weight",
            "degree_reward_weight",
            "agent_collision_distance",
            "collision_reward_weight",
            "obstacle_safety_margin",
            "obstacle_reward_weight",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(float(self.velocity_bonus_weight)):
            raise ValueError("velocity_bonus_weight must be finite")
        if self.position_target is not None and not isfinite(float(self.position_target)):
            raise ValueError("position_target must be finite when provided")
        if self.min_radius > self.max_radius:
            raise ValueError("min_radius cannot exceed max_radius")
        # A radius larger than L can connect across every coordinate and makes
        # the intended local stress test ambiguous.  Keep the audit boundary
        # explicit; users may still choose a large but finite radius <= L.
        if self.max_radius + self.contact_retention_margin > self.space_half_width:
            raise ValueError("max_radius must be <= space_half_width for strict locality")
        if self.flocking_normalization not in {"population", "local_mean"}:
            raise ValueError("flocking_normalization must be 'population' or 'local_mean'")
        for name in ("obstacle_radius", "obstacle_sensor_range"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        obstacles = tuple(_vector(center, self.dim) for center in self.obstacle_centers)
        if any(any(abs(value) > self.space_half_width for value in center) for center in obstacles):
            raise ValueError("obstacle centers must lie inside the periodic domain")
        object.__setattr__(self, "obstacle_centers", obstacles)


class DensityAdaptiveNeighborhood:
    """Local, density-adaptive radius and kernel calculator.

    The initial degree estimate uses ``base_radius``.  Each agent then gets
    ``r_i = base_radius * (target_degree / max(degree_i, 1))**radius_exponent``
    clipped to ``[min_radius, max_radius]``.  All subsequent operations only
    inspect pairs inside ``r_i``.  In particular, an empty local set stays
    empty; this class never samples distant agents as a fallback.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    @property
    def fallback_count(self) -> int:
        """Always zero; retained as an auditable counter-compatible property."""

        return 0

    def pair_delta(self, positions: Sequence[Sequence[Number]], i: int, j: int) -> Vector:
        return _toroidal_delta(
            positions[i], positions[j], self.config.space_half_width
        )

    def pair_distance_sq(
        self, positions: Sequence[Sequence[Number]], i: int, j: int
    ) -> float:
        return _norm_sq(self.pair_delta(positions, i, j))

    def base_degrees(self, positions: Sequence[Sequence[Number]]) -> tuple[int, ...]:
        n = len(positions)
        radius_sq = self.config.base_radius * self.config.base_radius
        result = []
        for i in range(n):
            count = 0
            for j in range(n):
                if i != j and self.pair_distance_sq(positions, i, j) <= radius_sq:
                    count += 1
            result.append(count)
        return tuple(result)

    def radii(
        self,
        positions: Sequence[Sequence[Number]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        """Return target radii, optionally smoothed from the previous step."""

        degrees = self.base_degrees(positions)
        cfg = self.config
        target = []
        for degree in degrees:
            scale = (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent
            target.append(_clip(cfg.base_radius * scale, cfg.min_radius, cfg.max_radius))
        target_radii = tuple(target)
        if previous_radii is None or cfg.radius_smoothing == 0.0:
            return target_radii
        previous = self.validate_radii(positions, previous_radii)
        alpha = cfg.radius_smoothing
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * target_radii[i],
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(target_radii))
        )

    def validate_radii(
        self, positions: Sequence[Sequence[Number]], radii: Sequence[float]
    ) -> tuple[float, ...]:
        """Validate and normalize a caller-supplied per-agent radius vector."""

        if len(radii) != len(positions):
            raise ValueError("radii length must match population size")
        normalized = tuple(float(radius) for radius in radii)
        if any(not isfinite(radius) or radius < 0.0 for radius in normalized):
            raise ValueError("radii must be finite and non-negative")
        return normalized

    def validate_graph(
        self, positions: Sequence[Sequence[Number]], graph: NeighborGraph
    ) -> NeighborGraph:
        """Validate a directed graph before it is used as a local observation."""

        n = len(positions)
        if len(graph) != n:
            raise ValueError("neighbor graph size must match population size")
        normalized: list[tuple[int, ...]] = []
        for i, local in enumerate(graph):
            seen: set[int] = set()
            row: list[int] = []
            for neighbor in local:
                if (
                    not isinstance(neighbor, Integral)
                    or isinstance(neighbor, bool)
                    or neighbor < 0
                    or neighbor >= n
                ):
                    raise ValueError("neighbor index out of range")
                neighbor = int(neighbor)
                if neighbor == i:
                    raise ValueError("neighbor graph cannot contain self-loops")
                if neighbor in seen:
                    raise ValueError("neighbor graph cannot contain duplicate indices")
                seen.add(neighbor)
                row.append(neighbor)
            normalized.append(tuple(row))
        return tuple(normalized)

    def neighbors(
        self,
        positions: Sequence[Sequence[Number]],
        radii: Optional[Sequence[float]] = None,
        previous_graph: Optional[NeighborGraph] = None,
    ) -> NeighborGraph:
        n = len(positions)
        if radii is None:
            radii = self.radii(positions)
        radii = self.validate_radii(positions, radii)
        if previous_graph is not None:
            previous_graph = self.validate_graph(positions, previous_graph)
        graph = []
        for i in range(n):
            radius_sq = float(radii[i]) ** 2
            local = [
                j
                for j in range(n)
                if i != j and self.pair_distance_sq(positions, i, j) <= radius_sq
            ]
            if previous_graph is not None and self.config.contact_retention_margin > 0.0:
                retention_radius = float(radii[i]) + self.config.contact_retention_margin
                retention_sq = retention_radius * retention_radius
                retained = [
                    j
                    for j in previous_graph[i]
                    if j not in local
                    and self.pair_distance_sq(positions, i, j) <= retention_sq
                ]
                local.extend(retained)
                local.sort()
            graph.append(tuple(local))
        return tuple(graph)

    def densities(
        self, positions: Sequence[Sequence[Number]], radii: Optional[Sequence[float]] = None
    ) -> tuple[float, ...]:
        """Return local number density using the adaptive neighbourhood."""

        if not positions:
            return tuple()
        if radii is None:
            used_radii = self.radii(positions)
        else:
            used_radii = self.validate_radii(positions, radii)
        graph = self.neighbors(positions, used_radii)
        dim = len(positions[0])
        # Volume of a unit d-ball.  For d=2 this is pi, which is the usual
        # flocking density convention.
        unit_ball_volume = pi ** (dim / 2.0) / _gamma_half_integer(dim)
        return tuple(
            len(graph[i])
            / (unit_ball_volume * max(float(used_radii[i]) ** dim, 1e-12))
            for i in range(len(graph))
        )

    def kernel_weights(
        self,
        positions: Sequence[Sequence[Number]],
        graph: Optional[NeighborGraph] = None,
    ) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Return `(neighbor_index, weight)` pairs with paper's kernel.

        Weights are normalized only over the strict local set.  If floating
        point underflow makes every weight zero, a uniform weight over those
        same local neighbours is used; this is not a global fallback.
        """

        if graph is None:
            graph = self.neighbors(positions)
        else:
            graph = self.validate_graph(positions, graph)
        result = []
        beta = self.config.beta
        for i, local in enumerate(graph):
            raw = []
            for j in local:
                distance_sq = self.pair_distance_sq(positions, i, j)
                if beta == 0:
                    weight = 1.0
                else:
                    # log1p is numerically stable for close neighbours.
                    weight = exp(-beta * log1p(distance_sq))
                raw.append((j, weight))
            total = sum(weight for _, weight in raw)
            if raw and total <= 0.0:
                uniform = 1.0 / len(raw)
                result.append(tuple((j, uniform) for j, _ in raw))
            elif raw:
                result.append(tuple((j, weight / total) for j, weight in raw))
            else:
                result.append(tuple())
        return tuple(result)


def _gamma_half_integer(dim: int) -> float:
    """Gamma(1 + d/2), evaluated without importing math.gamma for clarity."""

    # Volume denominator for integer dimensions up to the small dimensions
    # used in the prototype.  The recurrence handles both parity classes.
    if dim % 2 == 0:
        value = 1.0
        for k in range(1, dim // 2 + 1):
            value *= k
        return value
    # Gamma(m + 3/2) = (m+1/2)...(1/2)*sqrt(pi), where m=(dim-1)/2.
    value = sqrt(pi)
    for k in range(1, (dim + 1) // 2 + 1):
        value *= (k - 0.5)
    return value


class DeepSetsActor:
    """Shared local actor with permutation-invariant neighbour pooling.

    For each agent the actor receives only its own state and relative states of
    the supplied local neighbours.  A weighted sum/mean of neighbour features
    is passed through a tiny two-layer tanh MLP.  Because the pooling is a sum
    over a set and all parameters are shared, permuting agents only permutes
    the returned action list.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 32,
        action_limit: float = 1.0,
        velocity_scale: float = 1.0,
        space_scale: float = 1.0,
        seed: int = 0,
    ):
        if dim < 1 or hidden_dim < 1:
            raise ValueError("dim and hidden_dim must be positive")
        if action_limit <= 0 or velocity_scale <= 0 or space_scale <= 0:
            raise ValueError("action_limit, velocity_scale, and space_scale must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.action_limit = float(action_limit)
        self.velocity_scale = float(velocity_scale)
        self.space_scale = float(space_scale)
        # Feature: self position/velocity, pooled relative position/velocity,
        # mean distance, local degree fraction, radius, and has-neighbour bit.
        self.feature_dim = 4 * dim + 3
        rng = random.Random(seed)
        scale1 = 1.0 / sqrt(self.feature_dim)
        scale2 = 1.0 / sqrt(hidden_dim)
        self.w1 = tuple(
            tuple(rng.uniform(-scale1, scale1) for _ in range(self.feature_dim))
            for _ in range(hidden_dim)
        )
        self.b1 = tuple(0.0 for _ in range(hidden_dim))
        self.w2 = tuple(
            tuple(rng.uniform(-scale2, scale2) for _ in range(hidden_dim))
            for _ in range(dim)
        )
        self.b2 = tuple(0.0 for _ in range(dim))

    def _features(
        self,
        position: Sequence[Number],
        velocity: Sequence[Number],
        neighbor_relative_positions: Sequence[Sequence[Number]],
        neighbor_relative_velocities: Sequence[Sequence[Number]],
        radius: float,
        weights: Optional[Sequence[float]],
    ) -> Vector:
        dim = self.dim
        if len(position) != dim or len(velocity) != dim:
            raise ValueError("actor state dimension mismatch")
        if len(neighbor_relative_positions) != len(neighbor_relative_velocities):
            raise ValueError("relative position/velocity count mismatch")
        count = len(neighbor_relative_positions)
        if weights is None:
            normalized_weights = [1.0 / count] * count if count else []
        else:
            if len(weights) != count:
                raise ValueError("weight count mismatch")
            total = sum(max(float(weight), 0.0) for weight in weights)
            normalized_weights = (
                [max(float(weight), 0.0) / total for weight in weights]
                if count and total > 0
                else ([1.0 / count] * count if count else [])
            )

        # Absolute position is normalized by the map size rather than by the
        # current interaction radius.  Otherwise changing density changes the
        # apparent self-position feature and confounds the learned kernel.
        position_scale = max(self.space_scale, 1e-8)
        pooled_position = [0.0] * dim
        pooled_velocity = [0.0] * dim
        mean_distance = 0.0
        for rel_pos, rel_vel, weight in zip(
            neighbor_relative_positions,
            neighbor_relative_velocities,
            normalized_weights,
        ):
            if len(rel_pos) != dim or len(rel_vel) != dim:
                raise ValueError("relative feature dimension mismatch")
            for k in range(dim):
                pooled_position[k] += weight * float(rel_pos[k]) / position_scale
                pooled_velocity[k] += weight * float(rel_vel[k]) / self.velocity_scale
            mean_distance += weight * sqrt(_norm_sq(rel_pos)) / position_scale

        features = [float(value) / position_scale for value in position]
        features.extend(float(value) / self.velocity_scale for value in velocity)
        features.extend(pooled_position)
        features.extend(pooled_velocity)
        features.extend(
            [
                mean_distance,
                count / max(1.0, float(count + 1)),
                min(1.0, float(radius) / max(self.space_scale, 1e-8)),
            ]
        )
        return tuple(features)

    def act_local(
        self,
        position: Sequence[Number],
        velocity: Sequence[Number],
        neighbor_relative_positions: Sequence[Sequence[Number]],
        neighbor_relative_velocities: Sequence[Sequence[Number]],
        radius: float,
        weights: Optional[Sequence[float]] = None,
    ) -> Vector:
        """Act from one local observation; no population/global input exists."""

        features = self._features(
            position,
            velocity,
            neighbor_relative_positions,
            neighbor_relative_velocities,
            radius,
            weights,
        )
        hidden = tuple(
            tanh(_dot(row, features) + bias) for row, bias in zip(self.w1, self.b1)
        )
        return tuple(
            self.action_limit
            * tanh(_dot(row, hidden) + bias)
            for row, bias in zip(self.w2, self.b2)
        )

    def act_population(
        self,
        state: PopulationState,
        neighborhood: DensityAdaptiveNeighborhood,
        graph: Optional[NeighborGraph] = None,
        radii: Optional[Sequence[float]] = None,
    ) -> tuple[Vector, ...]:
        """Compute one shared-policy action per agent from local sets only."""

        if state.dim != self.dim:
            raise ValueError("state/actor dimension mismatch")
        if radii is None:
            radii = neighborhood.radii(state.positions)
        else:
            radii = neighborhood.validate_radii(state.positions, radii)
        if graph is None:
            graph = neighborhood.neighbors(state.positions, radii)
        else:
            graph = neighborhood.validate_graph(state.positions, graph)
        weights = neighborhood.kernel_weights(state.positions, graph)
        actions = []
        for i, local in enumerate(graph):
            rel_positions = [
                neighborhood.pair_delta(state.positions, i, j) for j in local
            ]
            rel_velocities = [
                _sub(state.velocities[i], state.velocities[j]) for j in local
            ]
            local_weights = [weight for _, weight in weights[i]]
            actions.append(
                self.act_local(
                    state.positions[i],
                    state.velocities[i],
                    rel_positions,
                    rel_velocities,
                    radii[i],
                    local_weights,
                )
            )
        if len(actions) != state.n_agents:
            raise RuntimeError("actor must return one action per agent")
        return tuple(actions)


@dataclass(frozen=True)
class StepResult:
    """Result of one simulator transition."""

    state: PopulationState
    actions: tuple[Vector, ...]
    rewards: tuple[float, ...]
    previous_neighbors: NeighborGraph
    neighbors: NeighborGraph
    previous_radii: tuple[float, ...]
    radii: tuple[float, ...]
    shaping: tuple[dict[str, float], ...]
    done: bool


@dataclass
class Rollout:
    """Finite trajectory plus local graph snapshots."""

    states: list[PopulationState] = field(default_factory=list)
    actions: list[tuple[Vector, ...]] = field(default_factory=list)
    rewards: list[tuple[float, ...]] = field(default_factory=list)
    neighbor_history: list[NeighborGraph] = field(default_factory=list)
    radius_history: list[tuple[float, ...]] = field(default_factory=list)
    safety_history: list[dict[str, float]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.actions)


class EALMFGSimulator:
    """Strict-local finite-horizon mean-field flocking environment."""

    def __init__(
        self,
        config: SimulationConfig = SimulationConfig(),
        initial_state: Optional[PopulationState] = None,
    ):
        self.config = config
        self.neighborhood = DensityAdaptiveNeighborhood(config)
        self.rng = random.Random(config.seed)
        self.state = initial_state or self._sample_initial_state(16)
        if self.state.dim != config.dim:
            raise ValueError("initial state dimension does not match config")
        # A no-argument reset starts the same episode again, including its
        # noise stream.  Explicit reset(state=...) or reset(n_agents=...) sets
        # a new episode baseline for subsequent paired rollouts.
        self._initial_state = self.state
        self._initial_rng_state = self.rng.getstate()
        self._radii_state: Optional[tuple[float, ...]] = None
        self._graph_cache_state: Optional[PopulationState] = None
        self._graph_cache: Optional[tuple[NeighborGraph, tuple[float, ...]]] = None
        self.t = 0

    def _sample_initial_state(self, n_agents: int) -> PopulationState:
        cfg = self.config
        if not cfg.obstacle_centers and cfg.agent_collision_distance == 0.0:
            positions = tuple(
                tuple(self.rng.uniform(-cfg.space_half_width, cfg.space_half_width) for _ in range(cfg.dim))
                for _ in range(n_agents)
            )
        else:
            sampled: list[Vector] = []
            for _ in range(n_agents):
                for _attempt in range(10_000):
                    candidate = tuple(
                        self.rng.uniform(-cfg.space_half_width, cfg.space_half_width)
                        for _ in range(cfg.dim)
                    )
                    obstacle_clear = all(
                        sqrt(_norm_sq(_toroidal_delta(candidate, center, cfg.space_half_width)))
                        >= cfg.obstacle_radius
                        for center in cfg.obstacle_centers
                    )
                    agent_clear = all(
                        sqrt(_norm_sq(_toroidal_delta(candidate, other, cfg.space_half_width)))
                        >= cfg.agent_collision_distance
                        for other in sampled
                    )
                    if obstacle_clear and agent_clear:
                        sampled.append(candidate)
                        break
                else:
                    raise RuntimeError("could not sample a collision-free initial population")
            positions = tuple(sampled)
        velocities = tuple(
            tuple(self.rng.uniform(-cfg.velocity_limit, cfg.velocity_limit) for _ in range(cfg.dim))
            for _ in range(n_agents)
        )
        return PopulationState(positions, velocities)

    def reset(self, state: Optional[PopulationState] = None, n_agents: Optional[int] = None) -> PopulationState:
        if state is not None:
            if state.dim != self.config.dim:
                raise ValueError("state dimension does not match config")
            self.state = state
            self._initial_state = state
            self._initial_rng_state = self.rng.getstate()
        elif n_agents is not None:
            if n_agents < 1:
                raise ValueError("n_agents must be positive")
            self.state = self._sample_initial_state(n_agents)
            self._initial_state = self.state
            self._initial_rng_state = self.rng.getstate()
        else:
            self.state = self._initial_state
            self.rng.setstate(self._initial_rng_state)
        self._radii_state = None
        self._graph_cache_state = None
        self._graph_cache = None
        self.t = 0
        return self.state

    def graph(self, state: Optional[PopulationState] = None) -> tuple[NeighborGraph, tuple[float, ...]]:
        current = state or self.state
        if current is self._graph_cache_state and self._graph_cache is not None:
            return self._graph_cache
        previous_radii = self._radii_state if current is self.state else None
        previous_graph = (
            self._graph_cache[0]
            if current is self.state and self._graph_cache is not None
            else None
        )
        radii = self.neighborhood.radii(current.positions, previous_radii)
        graph = self.neighborhood.neighbors(current.positions, radii, previous_graph)
        if current is self.state:
            self._radii_state = radii
            self._graph_cache_state = current
            self._graph_cache = (graph, radii)
        return graph, radii

    def reward_components(
        self,
        state: PopulationState,
        actions: Sequence[Sequence[Number]],
        graph: Optional[NeighborGraph] = None,
    ) -> tuple[dict[str, float], ...]:
        """Compute paper-style reward at ``(x_t, v_t, u_t)``.

        The flocking term is the empirical paper kernel, truncated to the
        strict local graph.  ``population`` normalization retains its 1/N
        factor; ``local_mean`` is provided for density-matched comparisons.
        """

        if len(actions) != state.n_agents:
            raise ValueError("one action is required per agent")
        action_vectors = tuple(_vector(action, state.dim) for action in actions)
        if graph is None:
            graph, _ = self.graph(state)
        kernel = self.neighborhood.kernel_weights(state.positions, graph)
        cfg = self.config
        result = []
        for i in range(state.n_agents):
            weighted_square_error = 0.0
            for j, normalized_weight in kernel[i]:
                # Recover the unnormalized paper kernel.  The normalized
                # weights are used only to make adaptive pooling stable; the
                # pairwise term itself remains explicit and auditable.
                d2 = self.neighborhood.pair_distance_sq(state.positions, i, j)
                paper_weight = exp(-cfg.beta * log1p(d2)) if cfg.beta else 1.0
                dv2 = _norm_sq(_sub(state.velocities[i], state.velocities[j]))
                weighted_square_error += paper_weight * dv2
            if cfg.flocking_normalization == "population":
                flocking = -weighted_square_error / state.n_agents
            elif graph[i]:
                flocking = -weighted_square_error / len(graph[i])
            else:
                flocking = 0.0

            control = -cfg.control_weight * _norm_sq(action_vectors[i])
            velocity_bonus = cfg.velocity_bonus_weight * max(
                abs(value) for value in state.velocities[i]
            )
            position_term = 0.0
            if cfg.position_target is not None and cfg.position_weight and state.dim >= 2:
                y = state.positions[i][1]
                target = float(cfg.position_target)
                distance = min(abs(y - target), abs(y + target))
                position_term = -cfg.position_weight * distance
            total = flocking + control + velocity_bonus + position_term
            result.append(
                {
                    "flocking": flocking,
                    "control": control,
                    "velocity_bonus": velocity_bonus,
                    "position": position_term,
                    "total": total,
                }
            )
        return tuple(result)

    def obstacle_delta(self, position: Sequence[Number], center: Sequence[Number]) -> Vector:
        """Shortest ``position - center`` displacement on the periodic domain."""

        return _toroidal_delta(position, center, self.config.space_half_width)

    def obstacle_clearance(self, position: Sequence[Number], center: Sequence[Number]) -> float:
        return sqrt(_norm_sq(self.obstacle_delta(position, center))) - self.config.obstacle_radius

    def safety_snapshot(self, state: Optional[PopulationState] = None) -> dict[str, float]:
        """Return state-level collision and obstacle diagnostics."""

        current = state or self.state
        cfg = self.config
        collided_agents = 0
        minimum_agent_distance = 2.0 * cfg.space_half_width
        for i in range(current.n_agents):
            collided = False
            for j in range(current.n_agents):
                if i == j:
                    continue
                distance = sqrt(self.neighborhood.pair_distance_sq(current.positions, i, j))
                minimum_agent_distance = min(minimum_agent_distance, distance)
                collided = collided or (
                    cfg.agent_collision_distance > 0.0
                    and distance < cfg.agent_collision_distance
                )
            collided_agents += int(collided)

        obstacle_collisions = 0
        obstacle_proximity = 0
        minimum_clearance = 2.0 * cfg.space_half_width
        for position in current.positions:
            clearances = [
                self.obstacle_clearance(position, center)
                for center in cfg.obstacle_centers
            ]
            if clearances:
                nearest = min(clearances)
                minimum_clearance = min(minimum_clearance, nearest)
                obstacle_collisions += int(nearest < 0.0)
                obstacle_proximity += int(nearest < cfg.obstacle_safety_margin)
        return {
            "agent_collision_rate": collided_agents / current.n_agents,
            "obstacle_collision_rate": obstacle_collisions / current.n_agents,
            "obstacle_proximity_rate": obstacle_proximity / current.n_agents,
            "minimum_agent_distance": minimum_agent_distance,
            "minimum_obstacle_clearance": minimum_clearance,
        }

    def transition_shaping(
        self,
        previous_graph: NeighborGraph,
        next_graph: NeighborGraph,
        next_state: PopulationState,
    ) -> tuple[dict[str, float], ...]:
        """Compute bounded local contact, degree, and safety shaping terms."""

        cfg = self.config
        target_degree = max(float(cfg.target_degree), 1.0)
        rows: list[dict[str, float]] = []
        for i, (before, after) in enumerate(zip(previous_graph, next_graph)):
            before_set = set(before)
            persistence = (
                len(before_set.intersection(after)) / len(before_set)
                if before_set
                else 0.0
            )
            degree_score = min(len(after) / target_degree, 1.0)
            contact_loss = 1.0 - persistence if before_set else 0.0
            degree_deficit = max(0.0, 1.0 - degree_score)
            contact_term = -cfg.contact_reward_weight * contact_loss
            degree_term = -cfg.degree_reward_weight * degree_deficit * degree_deficit

            collision_deficit = 0.0
            if cfg.agent_collision_distance > 0.0:
                nearest = min(
                    (
                        sqrt(self.neighborhood.pair_distance_sq(next_state.positions, i, j))
                        for j in range(next_state.n_agents)
                        if i != j
                    ),
                    default=cfg.agent_collision_distance,
                )
                collision_deficit = min(
                    1.0,
                    max(0.0, (cfg.agent_collision_distance - nearest) / cfg.agent_collision_distance),
                )
            collision_term = -cfg.collision_reward_weight * collision_deficit * collision_deficit

            obstacle_deficit = 0.0
            if cfg.obstacle_centers and cfg.obstacle_safety_margin > 0.0:
                nearest_clearance = min(
                    self.obstacle_clearance(next_state.positions[i], center)
                    for center in cfg.obstacle_centers
                )
                obstacle_deficit = min(
                    1.0,
                    max(
                        0.0,
                        (cfg.obstacle_safety_margin - nearest_clearance)
                        / cfg.obstacle_safety_margin,
                    ),
                )
            obstacle_term = -cfg.obstacle_reward_weight * obstacle_deficit * obstacle_deficit
            total = contact_term + degree_term + collision_term + obstacle_term
            rows.append(
                {
                    "contact": contact_term,
                    "degree": degree_term,
                    "collision": collision_term,
                    "obstacle": obstacle_term,
                    "total": total,
                }
            )
        return tuple(rows)

    def _wrap_position(self, value: float) -> float:
        """Fold one coordinate back into ``[-L, L)`` (periodic torus).

        Extracted from :meth:`step` so boundary-conditional subclasses
        (e.g. reflective arenas) can override it without duplicating the
        transition logic.
        """

        cfg = self.config
        period = 2.0 * cfg.space_half_width
        return ((value + cfg.space_half_width) % period) - cfg.space_half_width

    def step(
        self,
        actions: Sequence[Sequence[Number]],
        noise: Optional[Sequence[Sequence[Number]]] = None,
    ) -> StepResult:
        """Advance one step with the paper's old-velocity position update."""

        state = self.state
        if len(actions) != state.n_agents:
            raise ValueError("one action is required per agent")
        action_vectors = tuple(
            _clip_vector(_vector(action, state.dim), -self.config.acceleration_limit, self.config.acceleration_limit)
            for action in actions
        )
        previous_graph, previous_radii = self.graph(state)
        rewards = tuple(
            component["total"]
            for component in self.reward_components(state, action_vectors, previous_graph)
        )

        if noise is not None and len(noise) != state.n_agents:
            raise ValueError("noise must contain one vector per agent")
        next_positions = []
        next_velocities = []
        cfg = self.config
        for i in range(state.n_agents):
            old_position = state.positions[i]
            old_velocity = state.velocities[i]
            # Equation x_{t+1}=x_t+v_t dt: deliberately use old_velocity.
            moved = _add_scaled(old_position, old_velocity, cfg.dt)
            wrapped = tuple(self._wrap_position(value) for value in moved)
            if noise is None:
                noise_vector = tuple(
                    self.rng.gauss(0.0, cfg.noise_std) for _ in range(state.dim)
                )
            else:
                noise_vector = _vector(noise[i], state.dim)
            accelerated = tuple(
                old_velocity[k] + action_vectors[i][k] * cfg.dt + noise_vector[k]
                for k in range(state.dim)
            )
            clipped_velocity = _clip_vector(
                accelerated, -cfg.velocity_limit, cfg.velocity_limit
            )
            next_positions.append(wrapped)
            next_velocities.append(clipped_velocity)

        next_state = PopulationState(tuple(next_positions), tuple(next_velocities))
        self.state = next_state
        self.t += 1
        next_graph, next_radii = self.graph(next_state)
        shaping = self.transition_shaping(previous_graph, next_graph, next_state)
        rewards = tuple(
            reward + terms["total"]
            for reward, terms in zip(rewards, shaping)
        )
        return StepResult(
            state=next_state,
            actions=action_vectors,
            rewards=rewards,
            previous_neighbors=previous_graph,
            neighbors=next_graph,
            previous_radii=previous_radii,
            radii=next_radii,
            shaping=shaping,
            done=False,
        )


def run_rollout(
    simulator: EALMFGSimulator,
    actor: PopulationActor,
    steps: int,
    reset: bool = False,
) -> Rollout:
    """Run a finite local-policy rollout and retain all audit snapshots."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if reset:
        simulator.reset()
    rollout = Rollout(states=[simulator.state])
    graph, radii = simulator.graph()
    rollout.neighbor_history.append(graph)
    rollout.radius_history.append(radii)
    rollout.safety_history.append(simulator.safety_snapshot())
    for _ in range(steps):
        actions = actor.act_population(simulator.state, simulator.neighborhood, graph, radii)
        result = simulator.step(actions)
        rollout.actions.append(result.actions)
        rollout.rewards.append(result.rewards)
        rollout.states.append(result.state)
        rollout.neighbor_history.append(result.neighbors)
        rollout.radius_history.append(result.radii)
        rollout.safety_history.append(simulator.safety_snapshot(result.state))
        graph, radii = result.neighbors, result.radii
    return rollout
