from inspect import signature
from ...problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from typing import List


class UpperConfBound(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """
    The Upper Confidence Bound aquisition method

    Uses the methods described by
    "https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047"

    to select points for evaluation based on:
    the largest upper confidence bound 
    """

    def __init__(self, model, refit_model: bool = True):
        """
        Args:
            model: Scikit-learn model used to make inferences
        """
        super().__init__(model, refit_model)
        # Check if the function supports "return_std"
        if 'return_std' not in signature(self.model.predict).parameters:
            raise ValueError('The model must have "return_std" in the predict methods')

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int, epsilon=1):
        self.epsilon = epsilon
        if self.fit_model:
            self.model.fit(*problem.get_labeled_points())
        return super().select_points(problem, n_to_select)

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        y_mean, y_std = self.model.predict(problem.points[inds], return_std=True)
        return y_mean + (self.epsilon * y_std)
