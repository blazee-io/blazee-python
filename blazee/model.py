"""
blazee.model
============

This module provides the BlazeeModel and ModelVersion classes to interact with models
deployed on Blazee.

Usage:

    # To make a single prediction from the default version of the model
    >>> model.predict(X[0])

    # To make batch predictions from the default version of the model
    >>> model.batch_predict(X)

    # To makke a single prediction from a specific model version
    >>> version.predict(X[0])

    # To deploy a new version of the model
    >>> model.update(new_model, default=False)
"""
import logging
import time

import requests
from dateutil import parser

from blazee.keras_utils import is_keras, serialize_keras
from blazee.prediction import Prediction
from blazee.pytorch_utils import is_pytorch, serialize_pytorch
from blazee.sklearn_utils import is_sklearn, serialize_sklearn
from blazee.utils import generate_zip, pretty_size


def _serialize_model(model, include_files=None):
    if is_sklearn(model):
        return serialize_sklearn(model, include_files)
    elif is_keras(model):
        return serialize_keras(model, include_files)
    elif is_pytorch(model):
        return serialize_pytorch(model, include_files)
    else:
        raise TypeError(f'Model Type not supported: {type(model)}')


class BlazeeModel:
    """BlazeeModel represents models that are deployed on Blazee.
    """

    def __init__(self, client, response):
        self.client = client
        self._deleted = False
        self._reset(response)

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
        self._check_deleted()
        self.client._api_call(f'/models/{self.id}',
                              method='PATCH',
                              json={
                                  'name': name
                              })
        self._refresh()
        return self

    def versions(self):
        """Returns all the version of this model

        Returns
        -------
        versions: `list` of blazee.model.ModelVersion`
        """
        self._check_deleted()
        resp = self.client._api_call(f'/models/{self.id}/versions')
        return [ModelVersion(self.client, self, r) for r in resp]

    def get_version(self, name_or_id):
        """Returns the version of this model from its name or ID.

        Returns
        -------
        model_version: `blazee.model.ModelVersion`
        """
        self._check_deleted()
        for version in self.versions():
            if name_or_id in [version.id, version.name]:
                return version

        raise ValueError(f"No version with ID or name {name_or_id}")

    def delete(self):
        """Deletes this model from Blazee"""
        self._check_deleted()
        self.client._api_call(f'/models/{self.id}',
                              method='DELETE')
        self._deleted = True

    def update(self, model, default=False, include_files=None):
        """Creates a new version of this model on Blazee and deploys it.
        Once this is finished, and if the deployment succeeds,
        the new version can be used for predictions.
        By default, the new version WILL NOT be set as the default version of the model.
        Set `default=True` during the update, or call `version.make_default()` to make it the default.

        Parameters
        ----------
        model: one of `sklearn.base.BaseEstimator`, `keras.models.Model`, `torch.nn.Module`
            The model to deploy
        default: `bool`
            Whether or not the new model version should be set as default or not.
        include_files: `list` of `str`
            The list of python files this model depends on, if the model depends on
            custom python code.
            Those dependencies will be packaged and distributed with the model.

        Returns
        -------
        model: `blazee.model.BlazeeModel`
            The Blazee model that was deployed, ready to use for
            predictions
        """
        self._check_deleted()
        serialized_model = _serialize_model(model, include_files=include_files)

        version = self._upload_version(serialized_model)
        return version.model

    def predict(self, data):
        """Gets a single prediction from the default version of this model.

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
        self._check_deleted()
        resp = self.client._api_call(f'/models/{self.id}/predict',
                                     method='POST',
                                     json=data)
        return Prediction(resp)

    def predict_batch(self, data):
        """Gets a batch of predictions from the default version of this model.


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
        self._check_deleted()
        resp = self.client._api_call(f'/models/{self.id}/predict_batch',
                                     method='POST',
                                     json=data)

        return [Prediction(p) for p in resp]

    def _reset(self, response):
        self.id = response['id']
        self.name = response['name']
        if response['default_version']:
            self.default_version = ModelVersion(self.client,
                                                self,
                                                response['default_version'])
        else:
            self.default_version = None
        self.created_at = parser.parse(response['created_at'])
        self.updated_at = parser.parse(response['updated_at'])

    def _refresh(self):
        resp = self.client._api_call(f'/models/{self.id}')
        self._reset(resp)

    def _check_deleted(self):
        if self._deleted:
            raise AssertionError('Model is deleted. Cannot perform operation')

    def __repr__(self):
        return f"<BlazeeModel '{self.name}'\n\tid={self.id}>"

    def __str__(self):
        return f"<BlazeeModel '{self.name}'\n\tid={self.id}>"

    def _upload_version(self, serialized_model):
        resp = self.client._api_call(f'/models/{self.id}/versions',
                                     method="POST",
                                     json={
                                         'type': serialized_model.type,
                                         'meta': serialized_model.metadata
                                     })

        upload_data = resp['upload_data']
        content = generate_zip(serialized_model.files)
        logging.info(
            f'Uploading model version to Blazee  ({pretty_size(len(content))})...')
        upload_resp = requests.post(upload_data['url'],
                                    data=upload_data['fields'],
                                    files={'file': content})
        upload_resp.raise_for_status()

        version = ModelVersion(self.client, self, resp)
        logging.info(f"Deploying new model version: {version.name}...")
        self.client._api_call(
            f'/models/{self.id}/versions/{version.id}/deploy',
            method='PATCH')
        version = self._wait_until_deployed(version)
        logging.info(f"Successfully deployed model version {version.id}")
        self._refresh()
        return version

    def _wait_until_deployed(self, version, sleep=10, max_retries=30):
        left = max_retries
        while left:
            if version.deployed:
                return version
            if version.deployment_error:
                raise RuntimeError(
                    "An error occurred while deploying the model")
            time.sleep(sleep)
            resp = self.client._api_call(
                f'/models/{version.model.id}/versions/{version.id}')
            version = ModelVersion(self.client, self, resp)
            left -= 1
        raise TimeoutError("The model was not deployed")


class ModelVersion:
    """BlazeeModel represents a version of a model deployed on Blazee.
    """

    def __init__(self, client, model, response):
        self.client = client
        self.model = model
        self.id = response['id']
        self.name = response['name']
        self.type = response['type']
        self.meta = response['meta']
        self.deployed = response['deployed']
        self.deployment_error = response['deployment_error']
        self.created_at = parser.parse(response['created_at'])
        self.updated_at = parser.parse(response['updated_at'])

    def __repr__(self):
        return f"<ModelVersion '{self.model.name}' @ {self.name}\n\tid={self.id}\n\tdeployed={self.deployed}\n\tcreated_at={self.created_at}>"

    def __str__(self):
        return f"<ModelVersion '{self.model.name}' @ {self.name}\n\tid={self.id}\n\tdeployed={self.deployed}\n\tcreated_at={self.created_at}>"

    def predict(self, data):
        """Gets a single prediction from this model version

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
        resp = self.client._api_call(f'/models/{self.model.id}/versions/{self.id}/predict',
                                     method='POST',
                                     json=data)
        return Prediction(resp)

    def predict_batch(self, data):
        """Gets a batch of predictions from this model version

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
        resp = self.client._api_call(f'/models/{self.model.id}/versions/{self.id}/predict_batch',
                                     method='POST',
                                     json=data)

        return [Prediction(p) for p in resp]

    def make_default(self):
        """Makes this version the default version of the model.
        Once this version is the default, it will be used to compute
        predictions for this model.
        """
        self.client._api_call(f'/models/{self.model.id}',
                              method='PATCH',
                              json={
                                  'default_version_id': self.id
                              })
        self.model._refresh()
