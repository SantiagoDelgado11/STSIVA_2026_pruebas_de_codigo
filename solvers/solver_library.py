"""Action-to-solver mapping for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SolverLibrary:
    """Container that maps discrete actions to solver wrappers.

    Action space:
    0 -> DDNM
    1 -> DPS
    2 -> DiffPIR
    """

    diffpir_solver: object
    dps_solver: object
    ddnm_solver: object

    def __post_init__(self) -> None:
        self._action_to_solver = {
            0: self.ddnm_solver,
            1: self.dps_solver,
            2: self.diffpir_solver,
        }

    def ddnm_step(self, x_k: Any, y, Phi):
        """Apply DDNM update for the current estimate.

        x_k is accepted for interface compatibility with iterative dynamics.
        """
        _ = x_k
        return self.ddnm_solver.solve(y, Phi)

    def dps_step(self, x_k: Any, y, Phi):
        """Apply DPS update for the current estimate.

        x_k is accepted for interface compatibility with iterative dynamics.
        """
        _ = x_k
        return self.dps_solver.solve(y, Phi)

    def diffpir_step(self, x_k: Any, y, Phi):
        """Apply DiffPIR update for the current estimate.

        x_k is accepted for interface compatibility with iterative dynamics.
        """
        _ = x_k
        return self.diffpir_solver.solve(y, Phi)

    def apply_action(self, action: int, x_k, y, Phi):
        """Route an action to the corresponding diffusion reconstruction step."""
        if action == 0:
            return self.ddnm_step(x_k=x_k, y=y, Phi=Phi)
        if action == 1:
            return self.dps_step(x_k=x_k, y=y, Phi=Phi)
        if action == 2:
            return self.diffpir_step(x_k=x_k, y=y, Phi=Phi)
        raise ValueError(f"Invalid action {action}. Expected one of {list(self._action_to_solver)}")

    @property
    def action_dim(self) -> int:
        """Number of available discrete actions."""
        return len(self._action_to_solver)

    def get_solver(self, action: int):
        """Return solver associated with action index."""
        if action not in self._action_to_solver:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self._action_to_solver)}")
        return self._action_to_solver[action]
