from inspect import signature
import numpy as np
from ...problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from scipy.stats import norm
from typing import List


class ProbOfImprovement(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """
    The Probability of Improvement aquisition method

    Uses the methods described by
    `Eric Brochu, Vlad M. Cora and Nando de Freitas https://arxiv.org/pdf/1012.2599.pdf`_
    and `https://machinelearningmastery.com/what-is-bayesian-optimization/`

    to select points for evaluation based on:
    the largest probability of improvement

    For further understanding of a choice of epsilon:
    "In this case, the difficulty is that [the PI(·) method] is extremely sensitive 
    to the choice of the target.If the desired improvement is too small, the search 
    will be highly local and will only move on to search globally after searching nearly
    exhaustively around the current best point. On the other hand, if
    [ξ] is set too high, the search will be excessively global, and the
    algorithm will be slow to fine-tune any promising solutions."

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
        """
        Args:
            problem (dict): Active learning problem dictionary
            n_to_select (int): number of points to select from the unlabeled pool
            epsilon (int): how local/ global you want the search to be 
        """
        self.epsilon = epsilon
        if self.fit_model:
            self.model.fit(*problem.get_labeled_points())
        return super().select_points(problem, n_to_select)

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        _, known_labels = problem.get_labeled_points()
        threshold = np.max(known_labels)
        y_mean, y_std = self.model.predict(problem.points[inds], return_std=True)
        return norm.cdf((y_mean - threshold - self.epsilon) / y_std)