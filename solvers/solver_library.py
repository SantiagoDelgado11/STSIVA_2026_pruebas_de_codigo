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
        for action, solver in self._action_to_solver.items():
            if not callable(getattr(solver, "solve", None)):
                raise TypeError(
                    f"Solver mapped to action {action} ({type(solver).__name__}) "
                    "must implement a callable `solve` method."
                )


    def _validate_action(self, action: int) -> None:
        if action not in self._action_to_solver:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self._action_to_solver)}")

    def _solve(self, action: int, x_k: Any, y, Phi, ground_truth = None):
        solver = self.get_solver(action)
        return solver.solve(x_k=x_k, y=y, H=Phi, ground_truth=ground_truth)
 

    def ddnm_step(self, x_k: Any, y, Phi):
        return self._solve(action=0, x_k=x_k, y=y, Phi=Phi)


    def dps_step(self, x_k: Any, y, Phi):
        return self._solve(action=1, x_k=x_k, y=y, Phi=Phi)
       

    def diffpir_step(self, x_k: Any, y, Phi):
        return self._solve(action=2, x_k=x_k, y=y, Phi=Phi)

    def apply_action(self, action: int, x_k, y, Phi, ground_truth = None):
        """Route an action to the corresponding diffusion reconstruction step."""
        self._validate_action(action)
        return self._solve(action=action, x_k=x_k, y=y, Phi=Phi, ground_truth=ground_truth) 


    @property
    def action_dim(self) -> int:
        """Number of available discrete actions."""
        return len(self._action_to_solver)
    
    @property
    def supports_continuation(self) -> bool:
        """True only if every solver can continue from x_k instead of restarting from noise."""
        return all(bool(getattr(solver, "supports_continuation", False)) for solver in self._action_to_solver.values())

    @property
    def supports_continual_actions(self) -> bool:
        """Backward-compatible alias used by parts of the environment code."""
        return self.supports_continuation

    @property
    def is_contextual_bandit(self) -> bool:
        """When continuation is not supported, the interaction is contextual bandit (single-step)."""
        return not self.supports_continuation


    def get_solver(self, action: int):
        """Return solver associated with action index."""
        self._validate_action(action)
        return self._action_to_solver[action]
