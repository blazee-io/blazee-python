"""
blazee.prediction
=================

This module provides a Prediction object representing a single prediction
of a deployed Blazee Model.


"""


class Prediction:
    """Prediction object representing a single prediction
    of a deployed Blazee Model for a supervised learning problem.

    Attributes:
    -----------

    prediction: object
        For a classification problem, this will be the class predicted by
        the model.
        For a regression problem, this will be the predicted value.

    probas: list of `float`
        For a classification problem, this will be a list of the predicted
        probabilities for each class.
        For a regression problem, this will be `None`
    """

    def __init__(self, resp):
        self.prediction = resp['prediction']
        try:
            self.probas = resp['probas']
        except KeyError:
            self.probas = None

    def __repr__(self):
        return f"<Prediction\n\tprediction={self.prediction}\n\tprobas={self.probas}>"

    def __str__(self):
        return f"<Prediction\n\tprediction={self.prediction}\n\tprobas={self.probas}>"
