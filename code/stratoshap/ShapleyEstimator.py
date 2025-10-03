from abc import ABC, abstractmethod
import numpy as np

class ShapleyEstimator(ABC):
    def __init__(self):
        self.n = None
        self.budget = None
        self.game = None
        self.shapley_values = None
        self.player_steps = None

    def reset(self, game, budget):
        """Initializes the game and experiment parameters."""
        self.game = game
        self.n = self.game.number_of_players()
        self.budget = budget
        self.initial_budget = budget
        self.sv = np.zeros(self.n)
        self.player_steps = [0] * self.n

    @abstractmethod
    def approximate_shapley_values(self):
        """Must be implemented in subclasses to approximate Shapley values."""
        raise NotImplementedError("The approximate_shapley_values method must be overridden.")
    

    @abstractmethod
    def get_estimates(self):
        raise NotImplementedError


    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    def get_game_value(self, coalition):
        assert self.budget > 0, "Budget is exhausted."
        if len(coalition) == 0:
            if hasattr(self.game, 'dim'):
                return True, np.zeros((1, self.game.dim))
            else:
                return True, np.array([0]).reshape(1, -1)
        else:
            self.budget -= 1
            return self.budget > 0, self.game.compute_value(coalition)

    def update_shapley_value(self, player, estimate):
        """Updates the Shapley value for a given player based on a new estimate."""
        steps = self.player_steps[player]
        self.sv[player] = (self.sv[player] * steps + estimate) / (steps + 1)
        self.player_steps[player] += 1

        # print(f"shapley_values = {self.shapley_values}")

    def get_all_players(self):
        return list(range(self.n))