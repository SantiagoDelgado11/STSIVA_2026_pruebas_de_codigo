"""Action-to-solver mapping for RL training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolverLibrary:
    """Container that maps discrete actions to solver wrappers.

    Action space:
    0 -> DiffPIR
    1 -> DPS
    2 -> DDNM
    """

    diffpir_solver: object
    dps_solver: object
    ddnm_solver: object

    def __post_init__(self) -> None:
        self._action_to_solver = {
            0: self.diffpir_solver,
            1: self.dps_solver,
            2: self.ddnm_solver,
        }

    @property
    def action_dim(self) -> int:
        """Number of available discrete actions."""
        return len(self._action_to_solver)

    def get_solver(self, action: int):
        """Return solver associated with action index."""
        if action not in self._action_to_solver:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self._action_to_solver)}")
        return self._action_to_solver[action]
