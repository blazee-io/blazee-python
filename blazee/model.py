"""
blazee.model
============

This module provides the BlazeeModel to interact with models
deployed on Blazee.

Usage:

    >>> model.predict(X[0]) # To make a single prediction

    >>> model.batch_predict(X) # To make batch predictions
"""
from blazee.prediction import Prediction


class BlazeeModel:
    """BlazeeModel represents models that are deployed on Blazee.
    """

    def __init__(self, client, response):
        self.client = client
        self.id = response['id']
        self.name = response['name']
        self.type = response['type']
        self.data = response['data']
        self.uploaded = response['uploaded']

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self):
        return self.__repr__()

    def rename(self, name):
        """Renames the model on Blazee

        Parameters
        ----------
        name: string
            The name to set for the model

        Returns
        -------
        self: `blazee.model.BlazeeModel`
            The model
        """
        resp = self.client._api_call(f'/models/{self.id}',
                                     method='PATCH',
                                     json={
                                         'name': name
                                     })
        self.name = resp['name']
        return self

    def delete(self):
        """Deletes this model from Blazee"""
        self.client._api_call(f'/models/{self.id}',
                              method='DELETE')

    def predict(self, data):
        """Gets a single prediction from this model

        To make multiple predictions, see `batch_predict()`

        Parameters
        ----------

        data: list
            An array of features. This must be in the same format that what
            would be passed locally to the `predict()` method of the
            Scikit Learn model or pipeline.

        Returns
        -------
        prediction: `blazee.prediction.Prediction`
            The prediction
        """
        resp = self.client._api_call(f'/models/{self.id}/predict',
                                     method='POST',
                                     json=data)
        return Prediction(resp)

    def predict_batch(self, data):
        """Gets a batch of predictions from this model

        There is no size limit for the batch.
        To make a single predictions, see `predict()`

        Parameters
        ----------

        data: list of list
            A list of list of features. This must be in the same format
            that what would be passed locally to the `predict()` method
            of the Scikit Learn model or pipeline.

        Returns
        -------
        predictions: list of `blazee.prediction.Prediction`
            The predictions for the batch
        """
        resp = self.client._api_call(f'/models/{self.id}/predict_batch',
                                     method='POST',
                                     json=data)

        return [Prediction(p) for p in resp]
