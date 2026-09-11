"""First-round ICRA validation: analytical fixed-radius sweep and local-KNN.

Runs *training-free* controllers over three representative densities so the
core ALF claim can be tested against its two strongest reviewer objections:

1. P0-1: "ALF only wins because it is allowed a larger interaction radius."
   -> analytical fixed-radius sweep  r in {4, 5, 6, 7, 8}, using the *same*
   inverse-distance alignment law as ALF (beta=0.5, gain=1.0).  The paper's
   main-table fixed baseline is a learned SAC policy; this sweep isolates the
   neighbourhood mechanism.

2. P0-2: "A range-limited local-KNN would do the same."
   -> local-KNN with physical sensing cap r_s = 8.25 (the largest distance
   queried by ALF after its 0.25 contact-retention margin) and
   K in {4, 6, 8}; selection is strictly local (nearest K among visible
   neighbours, never beyond the sensing cap, no global statistics).

Both are compared against ALF (paper parameters: r_b=4, k=6, gamma=0.5,
smoothing 0.9, retention 0.25, interaction-radius cap min(8, L-0.25)).

All cells share: dynamics (x_{t+1}=x_t+v_t dt; v_{t+1}=clip(v_t+u_t dt)),
initial states (fixed evaluation seed bank), horizon, N=16, workspace,
velocity limit, acceleration limit, noise=0, and the alignment law.  Only the
neighbourhood mechanism differs.  The script is stdlib-only.

Usage::

    # Phase A smoke (1 seed, 20 steps, 1 density)
    python -m eal_mfg_icra.scripts.neighborhood_bench --mode smoke

    # Phase B quick validation (3 seeds x 200 steps x 3 densities)
    python -m eal_mfg_icra.scripts.neighborhood_bench --mode quick

    # explicit controls
    python -m eal_mfg_icra.scripts.neighborhood_bench --out experiments/next_icra_validation \
        --seeds 20000 20001 20002 --half-widths 9.0 12.0 18.0 --steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
from math import pi
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from ..core import (
    DensityAdaptiveNeighborhood,
    EALMFGSimulator,
    PopulationState,
    SimulationConfig,
    Vector,
    _clip,
    run_rollout,
)
from ..metrics import compute_connectivity_metrics, compute_metrics

# Paper ALF parameters (open scenario, see simulation_for in rl_train.py).
BASE_RADIUS = 4.0
TARGET_DEGREE = 6.0
RADIUS_EXPONENT = 0.5
SMOOTHING = 0.9
RETENTION_MARGIN = 0.25
MIN_RADIUS = 1.0
BETA = 0.5
PRIOR_GAIN = 1.0
DT = 0.5
VELOCITY_LIMIT = 1.0
ACCELERATION_LIMIT = 1.0
N_AGENTS = 16
NOISE_STD = 0.0
# ALF's nominal interaction-radius cap is 8.0.  Retention can query an old
# contact out to 8.25, so every method that performs candidate selection uses
# the same 8.25 physical sensing cap in the matched comparison.
INTERACTION_RADIUS_CAP = 8.0
SENSING_RANGE = INTERACTION_RADIUS_CAP + RETENTION_MARGIN

# Representative densities: high L=9 (0.04938), nominal L=12 (0.02778),
# low L=18 (0.01235); N=16 on the periodic square [-L, L)^2.
DEFAULT_HALF_WIDTHS = (9.0, 12.0, 18.0)
DEFAULT_SEEDS = (20_000, 20_001, 20_002)


def matched_half_width(n_agents: int, density: float) -> float:
    """Half-width L such that rho = n_agents / (2*L)^2 on the periodic square.

    The density-matched population-scaling protocol fixes rho and derives L
    from the population size, so that scaling the population does not change
    the operating density.
    """

    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    if density <= 0.0:
        raise ValueError("density must be positive")
    return 0.5 * (float(n_agents) / float(density)) ** 0.5


class AnalyticalAlignmentActor:
    """The paper's analytical alignment prior (training-free, strictly local).

    Reproduces ``ObservationBuilder._alignment_prior`` exactly: the control is
    ``clip(g * (weighted_local_velocity_target - v_i) / (a_max * dt), +-1)``
    with weights ``(1 + d^2)^(-beta)``.  An empty local set yields zero.
    """

    def __init__(
        self,
        beta: float = BETA,
        gain: float = PRIOR_GAIN,
        acceleration_limit: float = ACCELERATION_LIMIT,
        dt: float = DT,
    ) -> None:
        self.beta = float(beta)
        self.gain = float(gain)
        self.scale = max(float(acceleration_limit) * float(dt), 1e-6)

    def act_population(
        self,
        state: PopulationState,
        neighborhood: DensityAdaptiveNeighborhood,
        graph=None,
        radii=None,
    ) -> tuple[Vector, ...]:
        del radii
        dim = state.dim
        actions: list[Vector] = []
        for i in range(state.n_agents):
            local = graph[i] if graph is not None else ()
            if not local:
                actions.append((0.0,) * dim)
                continue
            weighted = [0.0] * dim
            total = 0.0
            for j in local:
                d2 = neighborhood.pair_distance_sq(state.positions, i, j)
                weight = 1.0 if self.beta == 0.0 else (1.0 + d2) ** (-self.beta)
                total += weight
                for k in range(dim):
                    weighted[k] += weight * state.velocities[j][k]
            if total <= 0.0:
                actions.append((0.0,) * dim)
                continue
            actions.append(
                tuple(
                    max(-1.0, min(1.0, self.gain * (weighted[k] / total - state.velocities[i][k]) / self.scale))
                    for k in range(dim)
                )
            )
        return tuple(actions)


class KNNNeighborhood(DensityAdaptiveNeighborhood):
    """Range-limited nearest-K neighbourhood (strictly local topology baseline).

    Agent ``i`` sees only pairs within the physical sensing range ``r_s`` and
    then keeps the nearest ``K`` by distance (ties broken by index); if fewer
    than ``K`` are visible it keeps all of them.  No global KNN, no global
    density, no connected-component information, no retention margin.
    """

    def __init__(self, config: SimulationConfig, k: int, sensing_range: float = SENSING_RANGE):
        super().__init__(config)
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.sensing_range = float(sensing_range)
        self.sensing_sq = self.sensing_range * self.sensing_range

    def neighbors(
        self,
        positions: Sequence[Sequence[float]],
        radii: Optional[Sequence[float]] = None,
        previous_graph=None,
    ) -> tuple[tuple[int, ...], ...]:
        del radii, previous_graph
        n_agents = len(positions)
        graph: list[tuple[int, ...]] = []
        for i in range(n_agents):
            candidates = []
            for j in range(n_agents):
                if i == j:
                    continue
                distance_sq = self.pair_distance_sq(positions, i, j)
                if distance_sq <= self.sensing_sq:
                    candidates.append((distance_sq, j))
            candidates.sort(key=lambda item: (item[0], item[1]))
            selected = [j for _, j in candidates[: self.k]]
            graph.append(tuple(selected))
        return tuple(graph)


class KNNSimulator(EALMFGSimulator):
    """Simulator whose neighbourhood module performs the local-KNN selection."""

    def __init__(self, config: SimulationConfig, k: int, sensing_range: float = SENSING_RANGE):
        super().__init__(config)
        self.neighborhood = KNNNeighborhood(config, k=k, sensing_range=sensing_range)


class FixedRadiusNeighborhood(DensityAdaptiveNeighborhood):
    """Fixed-radius policy with an explicit common candidate sensing cap.

    The interaction graph is still limited by the requested fixed radius.  The
    additional cap makes the physical candidate envelope auditable and keeps
    the fixed controls on the same ``r_s=8.25`` sensing budget as retained
    adaptive policies; it does not add any neighbours to the policy.
    """

    def __init__(self, config: SimulationConfig, sensing_range: float = SENSING_RANGE):
        super().__init__(config)
        if sensing_range <= 0.0:
            raise ValueError("sensing_range must be positive")
        self.sensing_range = float(sensing_range)
        self.sensing_sq = self.sensing_range * self.sensing_range

    def neighbors(
        self,
        positions: Sequence[Sequence[float]],
        radii: Optional[Sequence[float]] = None,
        previous_graph=None,
    ) -> tuple[tuple[int, ...], ...]:
        graph = super().neighbors(positions, radii, previous_graph)
        # The parent already limits fresh edges by the fixed radius and any
        # retained edge by radius + margin.  Keep an explicit post-filter so
        # the shared physical sensing envelope is enforced in this path too.
        return tuple(
            tuple(
                j
                for j in local
                if self.pair_distance_sq(positions, i, j) <= self.sensing_sq
            )
            for i, local in enumerate(graph)
        )


class CurrentRadiusDegreeNeighborhood(DensityAdaptiveNeighborhood):
    """Matched current-radius degree-feedback baseline (CRDF).

    ALF measures degree inside the fixed probe ``base_radius``.  CRDF keeps
    every other parameter unchanged but measures degree inside the previous
    realized interaction radius before issuing the next command.  At reset,
    both methods use the same base-radius initialization.  The simulator's
    normal smoothing and one-step retention path is then applied unchanged.
    """

    def current_degrees(
        self,
        positions: Sequence[Sequence[float]],
        radii: Sequence[float],
    ) -> tuple[int, ...]:
        used = self.validate_radii(positions, radii)
        result = []
        for i, radius in enumerate(used):
            radius_sq = radius * radius
            result.append(
                sum(
                    1
                    for j in range(len(positions))
                    if i != j and self.pair_distance_sq(positions, i, j) <= radius_sq
                )
            )
        return tuple(result)

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        degrees = (
            self.base_degrees(positions)
            if previous_radii is None
            else self.current_degrees(positions, previous_radii)
        )
        cfg = self.config
        targets = tuple(
            _clip(
                cfg.base_radius
                * (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent,
                cfg.min_radius,
                cfg.max_radius,
            )
            for degree in degrees
        )
        if previous_radii is None or cfg.radius_smoothing == 0.0:
            return targets
        previous = self.validate_radii(positions, previous_radii)
        alpha = cfg.radius_smoothing
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * targets[i],
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(targets))
        )


class CurrentRadiusMultiplicativeNeighborhood(CurrentRadiusDegreeNeighborhood):
    """Geometry-matched current-radius multiplicative feedback baseline.

    The degree is measured inside the previous realized interaction radius and
    the next unsmoothed command multiplies that radius by
    ``(target_degree / max(degree, 1))**radius_exponent``.  With exponent 1/2
    in two dimensions, the homogeneous plug-in model targets the requested
    degree in one unsmoothed update.  Reset uses the same fixed-probe command as
    ALF so that all adaptive controllers share an initial graph.
    """

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        cfg = self.config
        if previous_radii is None:
            return super().radii(positions, previous_radii=None)

        previous = self.validate_radii(positions, previous_radii)
        degrees = self.current_degrees(positions, previous)
        targets = tuple(
            _clip(
                previous[i]
                * (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent,
                cfg.min_radius,
                cfg.max_radius,
            )
            for i, degree in enumerate(degrees)
        )
        if cfg.radius_smoothing == 0.0:
            return targets
        alpha = cfg.radius_smoothing
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * targets[i],
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(targets))
        )


class FixedProbeCurrentBaseNeighborhood(DensityAdaptiveNeighborhood):
    """Factorial cross-condition: fixed measurement probe, current command base.

    FPCM retains ALF's fixed ``base_radius`` aperture when measuring occupancy,
    but uses the previous realized radius as the multiplicative command base.
    It is the missing cross-condition needed to distinguish an aperture effect
    from a stateful command-base effect.  Reset is deliberately identical to
    ALF so the four-factorial controllers begin from the same graph.
    """

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        if previous_radii is None:
            return super().radii(positions, previous_radii=None)

        cfg = self.config
        previous = self.validate_radii(positions, previous_radii)
        degrees = self.base_degrees(positions)
        targets = tuple(
            _clip(
                previous[i]
                * (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent,
                cfg.min_radius,
                cfg.max_radius,
            )
            for i, degree in enumerate(degrees)
        )
        if cfg.radius_smoothing == 0.0:
            return targets
        alpha = cfg.radius_smoothing
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * targets[i],
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(targets))
        )


class StarDisplayNeighborhood(CurrentRadiusDegreeNeighborhood):
    """Literature-aligned adaptive-range control from StarDisplay.

    Hildenbrandt, Carere, and Hemelrijk (Behavioral Ecology, 2010, Eq. 2)
    update each agent's metric interaction range as

    ``R_next = (1-s) R + s R_max (1 - |N(R)| / n_c)``.

    Here ``s = 1 - radius_smoothing`` and ``n_c = target_degree``.  The
    published law is embedded in the paper's shared finite-sensing comparison:
    the same radius bounds, alignment action, retention margin, initial state,
    and physical candidate envelope are used.  The target is clipped to the
    declared radius bounds when the affine expression is non-positive.
    """

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        cfg = self.config
        if previous_radii is None:
            # Keep the first graph matched across adaptive controllers; the
            # literature update is applied from the first closed-loop step.
            return DensityAdaptiveNeighborhood.radii(self, positions, None)

        previous = self.validate_radii(positions, previous_radii)
        degrees = self.current_degrees(positions, previous)
        targets = tuple(
            _clip(
                cfg.max_radius
                * (1.0 - float(degree) / float(cfg.target_degree)),
                cfg.min_radius,
                cfg.max_radius,
            )
            for degree in degrees
        )
        alpha = cfg.radius_smoothing
        if alpha == 0.0:
            return targets
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * targets[i],
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(targets))
        )


class GlobalDensityOracleNeighborhood(DensityAdaptiveNeighborhood):
    """Nonlocal ideal-density reference under the homogeneous disk model.

    After the shared adaptive-controller initialization, this diagnostic uses
    the true cell density to command ``sqrt(k / (pi rho))``.  It is explicitly
    not a decentralized baseline: its purpose is to show what global density
    alone can and cannot recover under the same clipping, smoothing, retention,
    dynamics, and alignment law.
    """

    def radii(
        self,
        positions: Sequence[Sequence[float]],
        previous_radii: Optional[Sequence[float]] = None,
    ) -> tuple[float, ...]:
        cfg = self.config
        if previous_radii is None:
            return super().radii(positions, None)
        previous = self.validate_radii(positions, previous_radii)
        density = len(positions) / (2.0 * cfg.space_half_width) ** cfg.dim
        if cfg.dim != 2:
            raise ValueError("global density oracle is defined for the paper's D=2 protocol")
        target = _clip(
            (cfg.target_degree / (pi * density)) ** 0.5,
            cfg.min_radius,
            cfg.max_radius,
        )
        if cfg.radius_smoothing == 0.0:
            return (target,) * len(positions)
        alpha = cfg.radius_smoothing
        return tuple(
            _clip(
                alpha * previous[i] + (1.0 - alpha) * target,
                cfg.min_radius,
                cfg.max_radius,
            )
            for i in range(len(positions))
        )


def _base_config(
    half_width: float,
    seed: int,
    *,
    noise_std: float = NOISE_STD,
    target_degree: float = TARGET_DEGREE,
    radius_exponent: float = RADIUS_EXPONENT,
    smoothing: float = SMOOTHING,
    max_radius_cap: float = INTERACTION_RADIUS_CAP,
) -> SimulationConfig:
    return SimulationConfig(
        dim=2,
        space_half_width=half_width,
        dt=DT,
        velocity_limit=VELOCITY_LIMIT,
        acceleration_limit=ACCELERATION_LIMIT,
        noise_std=noise_std,
        beta=BETA,
        base_radius=BASE_RADIUS,
        target_degree=target_degree,
        min_radius=MIN_RADIUS,
        max_radius=min(max_radius_cap, half_width - RETENTION_MARGIN),
        radius_exponent=radius_exponent,
        radius_smoothing=smoothing,
        contact_retention_margin=RETENTION_MARGIN,
        flocking_normalization="population",
        seed=seed,
    )


def _fixed_config(
    half_width: float,
    seed: int,
    radius: float,
    *,
    noise_std: float = NOISE_STD,
    contact_retention_margin: float = 0.0,
) -> SimulationConfig:
    # A radius larger than the workspace half-width would connect across the
    # torus and violate strict locality (core.py audit boundary), so the fixed
    # radius is capped at L.  The realized radius is recorded per cell, so a
    # capped cell is transparent in the output.
    cfg = _base_config(half_width, seed, noise_std=noise_std)
    retention = float(contact_retention_margin)
    if retention < 0.0:
        raise ValueError("contact_retention_margin must be non-negative")
    return SimulationConfig(
        **{
            **cfg.__dict__,
            "base_radius": float(radius),
            "radius_exponent": 0.0,
            "radius_smoothing": 0.0,
            "contact_retention_margin": retention,
            "max_radius": min(float(radius), float(half_width) - retention),
        }
    )


def _knn_config(
    half_width: float,
    seed: int,
    k: int,
    *,
    noise_std: float = NOISE_STD,
) -> SimulationConfig:
    cfg = _base_config(half_width, seed, noise_std=noise_std)
    return SimulationConfig(
        **{
            **cfg.__dict__,
            "base_radius": min(SENSING_RANGE, float(half_width)),
            "radius_exponent": 0.0,
            "radius_smoothing": 0.0,
            "contact_retention_margin": 0.0,
            "max_radius": min(SENSING_RANGE, float(half_width)),
        }
    )


def _method_label(method: str, parameter: float | int) -> str:
    if method == "fixed":
        return f"fixed_r{int(parameter)}"
    if method == "fixed_retained":
        return "fixed_r8_retained"
    if method == "knn":
        return f"knn_k{int(parameter)}"
    if method == "crdf":
        return "crdf"
    if method == "fpcm":
        return "fpcm"
    if method == "crmf":
        return "crmf"
    if method == "stardisplay":
        return "stardisplay"
    if method == "density_oracle":
        return "density_oracle"
    return method


def run_cell(
    method: str,
    parameter: float | int,
    half_width: float,
    seed: int,
    steps: int,
    *,
    n_agents: int = N_AGENTS,
    noise_std: float = NOISE_STD,
    target_degree: float = TARGET_DEGREE,
    radius_exponent: float = RADIUS_EXPONENT,
    smoothing: float = SMOOTHING,
    max_radius_cap: float = INTERACTION_RADIUS_CAP,
) -> dict[str, float | int | str]:
    """Run one (method, density, seed) cell and return per-episode metrics."""

    start = time.perf_counter()
    if method == "fixed":
        cfg = _fixed_config(half_width, seed, float(parameter), noise_std=noise_std)
        sim: EALMFGSimulator = EALMFGSimulator(cfg)
        sim.neighborhood = FixedRadiusNeighborhood(cfg)
    elif method == "fixed_retained":
        # Strict control for the maximum-radius comparison: fixed radius and
        # the same retention margin as ALF, with the same local cap boundary.
        cfg = _fixed_config(
            half_width,
            seed,
            float(parameter),
            noise_std=noise_std,
            contact_retention_margin=RETENTION_MARGIN,
        )
        sim: EALMFGSimulator = EALMFGSimulator(cfg)
        sim.neighborhood = FixedRadiusNeighborhood(cfg)
    elif method == "knn":
        cfg = _knn_config(half_width, seed, int(parameter), noise_std=noise_std)
        sim = KNNSimulator(cfg, k=int(parameter))
    elif method == "crdf":
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
        sim.neighborhood = CurrentRadiusDegreeNeighborhood(cfg)
    elif method == "crmf":
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
        sim.neighborhood = CurrentRadiusMultiplicativeNeighborhood(cfg)
    elif method == "fpcm":
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
        sim.neighborhood = FixedProbeCurrentBaseNeighborhood(cfg)
    elif method == "stardisplay":
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
        sim.neighborhood = StarDisplayNeighborhood(cfg)
    elif method == "density_oracle":
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
        sim.neighborhood = GlobalDensityOracleNeighborhood(cfg)
    else:
        cfg = _base_config(
            half_width,
            seed,
            noise_std=noise_std,
            target_degree=target_degree,
            radius_exponent=radius_exponent,
            smoothing=smoothing,
            max_radius_cap=max_radius_cap,
        )
        sim = EALMFGSimulator(cfg)
    state = sim.reset(n_agents=n_agents)
    actor = AnalyticalAlignmentActor()
    rollout = run_rollout(sim, actor, steps=steps)
    metrics = compute_metrics(rollout)
    connectivity = compute_connectivity_metrics(rollout)
    energy = 0.0
    for actions in rollout.actions:
        energy += sum(value * value for action in actions for value in action)
    # Reconstruct the aperture used by the feedback signal for every recorded
    # snapshot.  The command at t=0 is initialized from the fixed probe for
    # both adaptive methods; CRDF then uses the realized radius from t-1.
    measurement_degrees: list[tuple[int, ...]] = []
    command_history: list[tuple[float, ...]] = []
    for t, snapshot in enumerate(rollout.states):
        if method in {"crdf", "crmf", "stardisplay"} and t > 0:
            degrees = sim.neighborhood.current_degrees(
                snapshot.positions, rollout.radius_history[t - 1]
            )
        elif method in {"alf", "crdf", "fpcm", "crmf", "stardisplay"}:
            degrees = sim.neighborhood.base_degrees(snapshot.positions)
        else:
            # Fixed-radius and KNN controls have no radius feedback aperture;
            # use their realized graph degree as the matched diagnostic.
            degrees = tuple(len(local) for local in rollout.neighbor_history[t])
        measurement_degrees.append(tuple(int(value) for value in degrees))
        if method in {"alf", "crdf", "fpcm", "crmf", "stardisplay", "density_oracle"}:
            if method == "density_oracle" and t > 0:
                density = n_agents / (2.0 * half_width) ** 2
                oracle_target = _clip(
                    (cfg.target_degree / (pi * density)) ** 0.5,
                    cfg.min_radius,
                    cfg.max_radius,
                )
                command_history.append((oracle_target,) * n_agents)
                continue
            if method == "stardisplay" and t > 0:
                target = tuple(
                    _clip(
                        cfg.max_radius
                        * (1.0 - float(degree) / float(cfg.target_degree)),
                        cfg.min_radius,
                        cfg.max_radius,
                    )
                    for degree in degrees
                )
                command_history.append(target)
                continue
            command_base = (
                rollout.radius_history[t - 1]
                if method in {"fpcm", "crmf"} and t > 0
                else (cfg.base_radius,) * n_agents
            )
            target = tuple(
                _clip(
                    float(command_base[i])
                    * (cfg.target_degree / max(float(degree), 1.0)) ** cfg.radius_exponent,
                    cfg.min_radius,
                    cfg.max_radius,
                )
                for i, degree in enumerate(degrees)
            )
            command_history.append(target)
        else:
            command_history.append(tuple(float(value) for value in rollout.radius_history[t]))
    probe_degree = sum(sum(degrees) for degrees in measurement_degrees)
    radius_updates = [
        float(rollout.radius_history[t + 1][i]) - float(rollout.radius_history[t][i])
        for t in range(len(rollout.radius_history) - 1)
        for i in range(n_agents)
    ]
    abs_updates = [abs(value) for value in radius_updates]
    command_updates = [
        float(command_history[t + 1][i]) - float(command_history[t][i])
        for t in range(len(command_history) - 1)
        for i in range(n_agents)
    ]
    abs_command_updates = [abs(value) for value in command_updates]
    radius_update_rms = (
        sum(value * value for value in radius_updates) / len(radius_updates)
    ) ** 0.5 if radius_updates else 0.0
    reversal_pairs = 0
    signed_pairs = 0
    # Compare successive updates for the same agent only.  The history is
    # stored as (time, agent); comparing the flattened list would mix agents.
    for i in range(n_agents):
        agent_updates = [
            float(rollout.radius_history[t + 1][i])
            - float(rollout.radius_history[t][i])
            for t in range(len(rollout.radius_history) - 1)
        ]
        for previous, current in zip(agent_updates, agent_updates[1:]):
            if previous == 0.0 or current == 0.0:
                continue
            signed_pairs += 1
            reversal_pairs += int((previous > 0.0) != (current > 0.0))
    radius_transition_count = len(radius_updates)
    temporal_radius_std = []
    temporal_degree_var = []
    for i in range(n_agents):
        radius_series = [float(snapshot[i]) for snapshot in rollout.radius_history]
        radius_mean = statistics.fmean(radius_series)
        temporal_radius_std.append(
            (statistics.fmean((value - radius_mean) ** 2 for value in radius_series)) ** 0.5
        )
        degree_series = [len(graph[i]) for graph in rollout.neighbor_history]
        temporal_degree_var.append(statistics.pvariance(degree_series))
    min_radius = float(cfg.min_radius)
    max_radius = float(cfg.max_radius)
    radius_values = [float(value) for snapshot in rollout.radius_history for value in snapshot]
    degree_errors = [
        abs(float(degree) - float(cfg.target_degree))
        for degrees in measurement_degrees
        for degree in degrees
    ]
    degree_errors_sq = [value * value for value in degree_errors]
    interaction_area = pi * float(connectivity.get("mean_radius_sq", 0.0))
    row: dict[str, float | int | str] = {
        "method": _method_label(method, parameter),
        "mechanism": method,
        "parameter": float(parameter),
        "half_width": half_width,
        "density": n_agents / (2.0 * half_width) ** 2,
        "seed": seed,
        "steps": steps,
        "n_agents": n_agents,
        "noise_std": noise_std,
        "control_energy": energy / n_agents / steps,
        "mean_probe_degree": probe_degree / len(rollout.states) / n_agents,
        "mean_abs_radius_update": statistics.fmean(abs_updates) if abs_updates else 0.0,
        "radius_temporal_variation": statistics.fmean(abs_updates) if abs_updates else 0.0,
        "mean_abs_radius_command_update": statistics.fmean(abs_command_updates) if abs_command_updates else 0.0,
        "radius_command_temporal_variation": statistics.fmean(abs_command_updates) if abs_command_updates else 0.0,
        "rms_radius_update": radius_update_rms,
        "mean_agent_radius_temporal_std": statistics.fmean(temporal_radius_std),
        "mean_agent_degree_temporal_variance": statistics.fmean(temporal_degree_var),
        "radius_min_saturation_fraction": sum(value <= min_radius + 1e-12 for value in radius_values) / max(len(radius_values), 1),
        "radius_max_saturation_fraction": sum(value >= max_radius - 1e-12 for value in radius_values) / max(len(radius_values), 1),
        "radius_sign_reversal_rate": reversal_pairs / max(signed_pairs, 1),
        "radius_oscillation_frequency": reversal_pairs / max(radius_transition_count, 1),
        "radius_sign_reversal_count": reversal_pairs,
        "mean_degree_error": statistics.fmean(degree_errors) if degree_errors else 0.0,
        "mean_degree_error_sq": statistics.fmean(degree_errors_sq) if degree_errors_sq else 0.0,
        "mean_interaction_area": interaction_area,
        "visible_neighbor_count_mean": statistics.fmean(
            sum(
                1
                for j in range(n_agents)
                if i != j and sim.neighborhood.pair_distance_sq(snapshot.positions, i, j) <= SENSING_RANGE * SENSING_RANGE
            )
            for snapshot in rollout.states
            for i in range(n_agents)
        ),
        "distance_evaluations_per_step": n_agents * (n_agents - 1),
        "sorting_operations_per_step": n_agents if method == "knn" else 0,
        "runtime_s": time.perf_counter() - start,
    }
    row.update(metrics)
    row.update(connectivity)
    return row


_REPORT_KEYS = (
    "final_fragmentation",
    "mean_fragmentation",
    "final_largest_scc_fraction",
    "mean_largest_scc_fraction",
    "final_sym_lambda2",
    "polar_order",
    "velocity_dispersion",
    "neighbor_loss_rate",
    "contact_break_rate",
    "edge_persistence",
    "mean_degree",
    "final_mean_in_degree",
    "final_min_out_degree",
    "final_min_in_degree",
    "mean_edge_count",
    "mean_normalized_comm_cost",
    "mean_radius",
    "mean_radius_sq",
    "mean_interaction_area",
    "control_energy",
    "mean_probe_degree",
    "mean_abs_radius_update",
    "radius_temporal_variation",
    "mean_abs_radius_command_update",
    "radius_command_temporal_variation",
    "rms_radius_update",
    "mean_agent_radius_temporal_std",
    "mean_agent_degree_temporal_variance",
    "radius_min_saturation_fraction",
    "radius_max_saturation_fraction",
    "radius_sign_reversal_rate",
    "radius_oscillation_frequency",
    "radius_sign_reversal_count",
    "mean_degree_error",
    "mean_degree_error_sq",
    "visible_neighbor_count_mean",
    "distance_evaluations_per_step",
    "sorting_operations_per_step",
    "final_rooted_fraction",
)


def aggregate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-seed records into (cell, mean, std, 95% CI) summary rows."""

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["method"]), float(record["half_width"])), []).append(record)
    rows: list[dict[str, Any]] = []
    for (method, half_width), members in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method,
            "half_width": half_width,
            "density": members[0]["density"],
            "n_seeds": len(members),
        }
        for key in _REPORT_KEYS:
            values = [float(member[key]) for member in members if key in member]
            if values:
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                # Normal-approximation 95% CI (n>=30 in the formal protocol).
                row[f"{key}_ci95_low"] = mean - 1.96 * std / len(values) ** 0.5
                row[f"{key}_ci95_high"] = mean + 1.96 * std / len(values) ** 0.5
        rows.append(row)
    return rows


def _short(key: str) -> str:
    aliases = {
        "final_fragmentation": "finFrag",
        "mean_fragmentation": "mnFrag",
        "final_largest_scc_fraction": "SCC",
        "final_sym_lambda2": "lam2",
        "polar_order": "polar",
        "velocity_dispersion": "vdisp",
        "neighbor_loss_rate": "NLR",
        "contact_break_rate": "CBR",
        "edge_persistence": "persist",
        "mean_degree": "degree",
        "mean_edge_count": "edges",
        "mean_normalized_comm_cost": "comm",
        "mean_radius": "radius",
        "mean_radius_sq": "rSq",
        "control_energy": "energy",
        "mean_probe_degree": "probeDeg",
    }
    return aliases.get(key, key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "quick", "formal"], default="quick")
    parser.add_argument("--out", type=Path, default=Path("experiments/next_icra_validation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--half-widths", type=float, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-agents", type=int, default=N_AGENTS)
    parser.add_argument("--noise", type=float, default=NOISE_STD, help="actuation noise std")
    parser.add_argument("--k", type=float, default=TARGET_DEGREE, help="ALF target degree")
    parser.add_argument("--gamma", type=float, default=RADIUS_EXPONENT, help="ALF radius exponent")
    parser.add_argument("--alpha", type=float, default=SMOOTHING, help="ALF radius smoothing")
    parser.add_argument("--rmax", type=float, default=INTERACTION_RADIUS_CAP, help="adaptive interaction-radius cap")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="cell specs: fixed5, knn6, crdf, crmf, alf (default: full matched grid)",
    )
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    if args.mode == "smoke":
        seeds = seeds[:1]
        steps = args.steps or 20
        half_widths = tuple(args.half_widths or (12.0,))
    elif args.mode == "quick":
        steps = args.steps or 200
        half_widths = tuple(args.half_widths or DEFAULT_HALF_WIDTHS)
    else:  # formal: paper density grid x the full 50-seed evaluation bank
        if not args.seeds:
            seed_file = Path(__file__).resolve().parents[1] / "configs" / "eval_seeds.txt"
            seeds = tuple(
                int(raw.strip())
                for raw in seed_file.read_text(encoding="utf-8").splitlines()
                if raw.strip() and not raw.strip().startswith("#")
            )
        steps = args.steps or 200
        half_widths = tuple(args.half_widths or (7.0, 9.0, 11.0, 14.0, 18.0, 23.0, 30.0))

    if args.methods:
        cells: list[tuple[str, float | int]] = []
        for spec in args.methods:
            if spec == "alf":
                cells.append(("alf", 0.0))
            elif spec == "crdf":
                cells.append(("crdf", 0.0))
            elif spec == "crmf":
                cells.append(("crmf", 0.0))
            elif spec == "fixed_rmax_retained":
                cells.append(("fixed_retained", INTERACTION_RADIUS_CAP))
            elif spec.startswith("fixed"):
                cells.append(("fixed", int(spec.removeprefix("fixed"))))
            elif spec.startswith("knn"):
                cells.append(("knn", int(spec.removeprefix("knn"))))
            else:
                parser.error(f"unknown cell spec: {spec}")
    else:
        cells = [
            ("alf", 0.0),
            ("crdf", 0.0),
            ("crmf", 0.0),
            ("fixed_retained", INTERACTION_RADIUS_CAP),
        ] + [("fixed", r) for r in (4, 5, 6, 7, 8)] + [("knn", k) for k in (4, 6, 8)]

    records: list[dict[str, Any]] = []
    # Incremental checkpoint: one JSON line per completed cell, so a crashed
    # formal run can resume instead of restarting from scratch.
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "neighborhood_bench_checkpoint.jsonl"
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        print(f"resuming from checkpoint with {len(records)} completed cells", flush=True)
    completed = {(str(r["method"]), float(r["half_width"]), int(r["seed"])) for r in records}
    pending = 0
    for mechanism, parameter in cells:
        for half_width in half_widths:
            for seed in seeds:
                method = _method_label(mechanism, parameter)
                key = (method, half_width, seed)
                if key in completed:
                    continue
                pending += 1
                row = run_cell(
                    mechanism,
                    parameter,
                    half_width,
                    seed,
                    steps,
                    n_agents=args.n_agents,
                    noise_std=args.noise,
                    target_degree=args.k,
                    radius_exponent=args.gamma,
                    smoothing=args.alpha,
                    max_radius_cap=args.rmax,
                )
                records.append(row)
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                label = f"{row['method']} L={half_width:g} seed={seed}"
                print(
                    f"{label:<28} frag={row['final_fragmentation']:.3f} "
                    f"polar={row['polar_order']:.3f} deg={row['mean_degree']:.2f} "
                    f"r={row['mean_radius']:.2f}",
                    flush=True,
                )
    print(f"completed {len(records)} cells ({pending} new this run)", flush=True)

    summary_rows = aggregate(records)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "generated": True,
        "protocol": {
            "n_agents": args.n_agents,
            "noise_std": args.noise,
            "steps": steps,
            "seeds": list(seeds),
            "half_widths": list(half_widths),
            "densities": [args.n_agents / (2.0 * half_width) ** 2 for half_width in half_widths],
            "alignment": {"beta": BETA, "prior_gain": PRIOR_GAIN, "dt": DT},
            "alf": {"base_radius": BASE_RADIUS, "target_degree": args.k,
                    "radius_exponent": args.gamma, "smoothing": args.alpha,
                    "retention_margin": RETENTION_MARGIN, "max_radius_cap": args.rmax},
            "crmf": {"measurement_aperture": "previous realized radius",
                     "multiplicative_exponent": args.gamma},
            "knn": {"sensing_range": SENSING_RANGE, "k_values": [4, 6, 8]},
            "fixed": {"radii": [4, 5, 6, 7, 8], "sensing_range": SENSING_RANGE},
        },
        "cells": summary_rows,
        "episodes": records,
    }
    json_path = args.out / "neighborhood_bench.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in records for key in row})
    csv_path = args.out / "neighborhood_bench_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.out / "neighborhood_bench_summary.csv"
    summary_fields = ["method", "half_width", "density", "n_seeds"]
    for key in _REPORT_KEYS:
        summary_fields.extend((f"{key}_mean", f"{key}_std", f"{key}_ci95_low", f"{key}_ci95_high"))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in summary_fields})

    header = f"{'cell':<12} {'L':>5} {'dens':>8} |" + "".join(f"{_short(k):>8}" for k in
        ("final_fragmentation", "polar_order", "mean_degree", "mean_radius", "mean_normalized_comm_cost", "control_energy"))
    print("\n" + header)
    for row in summary_rows:
        line = f"{row['method']:<12} {row['half_width']:>5.1f} {row['density']:>8.4f} |"
        for key in ("final_fragmentation", "polar_order", "mean_degree", "mean_radius", "mean_normalized_comm_cost", "control_energy"):
            line += f"{row.get(key + '_mean', float('nan')):>8.3f}"
        print(line)
    print(f"\nrecords -> {json_path}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
