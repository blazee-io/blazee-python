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
import enum
import io
import logging
import pickle
import uuid
import zipfile

import requests
from dateutil import parser
from sklearn.pipeline import Pipeline

from blazee.prediction import Prediction


class ModelType(enum.Enum):
    SKLEARN = 'sklearn'


def _generate_zip(file_name, content):
    zipped = io.BytesIO()

    with zipfile.ZipFile(zipped, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zinfo = zipfile.ZipInfo(file_name)
        zinfo.external_attr = 0o644 << 16  # give read access
        zf.writestr(zinfo, content)

    return zipped.getvalue()


def _serialize_model(model):
    if _is_sklearn(model):
        if not type(model).__module__.startswith('sklearn.'):
            raise TypeError(
                f'Model of type {type(model)} is not supported. You must use a model or Pipeline that only contains estimators from the Scikit Learn library.')

        if isinstance(model, Pipeline):
            for name, step in model.steps:
                if not type(step).__module__.startswith('sklearn.'):
                    raise TypeError(
                        f'Pipeline contains step {name} of type {type(step)} which is not supported. You must use a model or Pipeline that only contains estimators from the Scikit Learn library.')

        # Check if model is fitted
        try:
            model.predict([0])
        except Exception as e:
            from sklearn.exceptions import NotFittedError
            if isinstance(e, NotFittedError):
                raise AttributeError("This model hasn't been trained yet")

        content = pickle.dumps(model)
        zipped = _generate_zip('model.pickle', content)
        return 'sklearn', zipped
    else:
        raise TypeError(f'Model Type not supported: {type(model)}')


def _get_model_metadata(model):
    if _is_sklearn(model):
        import sklearn
        return {
            'lib_versions': {
                'scikit-learn': sklearn.__version__
            }
        }
    else:
        raise TypeError(f'Model Type not supported: {type(model)}')


def _is_sklearn(model):
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)
    except:
        return False


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

    def update(self, model, default=False):
        """Creates a new version of this model on Blazee and deploys it.
        Once this is finished, and if the deployment succeeds,
        the new version can be used for predictions.
        By default, the new version WILL NOT be set as the default version of the model.
        Set `default=True` during the update, or call `version.make_default()` to make it the default.

        Parameters
        ----------
        model: `sklearn.base.BaseEstimator`
            The model to deploy
        default: `bool`
            Whether or not the new model version should be set as default or not.

        Returns
        -------
        model: `blazee.model.BlazeeModel`
            The Blazee model that was deployed, ready to use for
            predictions
        """
        self._check_deleted()
        model_type, content = _serialize_model(model)
        metadata = _get_model_metadata(model)

        version = self._upload_version(model_type, content, metadata)
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

    def _upload_version(self, model_type, content, metadata):
        resp = self.client._api_call(f'/models/{self.id}/versions',
                                     method="POST",
                                     json={
                                         'type': model_type,
                                         'meta': metadata
                                     })

        upload_data = resp['upload_data']
        logging.info(f'Uploading model version to Blazee...')
        upload_resp = requests.post(upload_data['url'],
                                    data=upload_data['fields'],
                                    files={'file': content})
        upload_resp.raise_for_status()

        version = ModelVersion(self.client, self, resp)
        logging.info(f"Deploying new model version: {version.name}...")
        deploy_resp = self.client._api_call(
            f'/models/{self.id}/versions/{version.id}/deploy',
            method='PATCH')
        logging.info(f"Successfully deployed model version {version.id}")
        self._refresh()
        return ModelVersion(self.client, self, deploy_resp)


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
