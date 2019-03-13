"""
blazee.client
=============

This module provides a Client for interacting with the Blazee API.
"""
import io
import pickle
import uuid
from datetime import datetime
from json import dumps as jsondumps

import requests
from requests.exceptions import HTTPError
from sklearn.base import BaseEstimator

from blazee.model import BlazeeModel
from blazee.utils import NumpyEncoder

BLAZEE_HOST = 'https://api.blazee.io/v1'


class Client:
    """Client for interacting with the Blazee API.

    Parameters
    ----------
    api_key : string
        Your Blazee API key. Get one at https://blazee.io
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('API Key must not be empty')
        self.api_key = api_key

    def all_models(self):
        """Returns a list of all your deployed Blazee models.

        Returns
        -------
        models: list of `blazee.model.BlazeeModel`
        """
        resp = self._api_call('/models')

        return [BlazeeModel(self, m) for m in resp]

    def get_model(self, model_id):
        """Returns the Blazee model with the give ID.

        Returns
        -------
        model: `blazee.model.BlazeeModel`
        """
        try:
            uuid.UUID(model_id)
        except ValueError:
            raise ValueError(f'Malformed model ID: {model_id}')

        resp = self._api_call(f'/models/{model_id}')

        return BlazeeModel(self, resp)

    def deploy_model(self, model, model_name=None):
        """Deploys a trained ML model on Blazee
        At the moment only Scikit Learn models and pipelines are
        supported.
        Using another framework? Reach out at support@blazee.io

        Parameters
        ----------
        model: `sklearn.base.BaseEstimator`
            The model to deploy
        model_name: string
            A custom name for the Blazee model

        Returns
        -------
        model: `blazee.model.BlazeeModel`
            The Blazee model that was deployed, ready to use for
            predictions
        """
        if isinstance(model, BaseEstimator):
            return self.deploy_sklearn(model, model_name)
        else:
            raise TypeError(f'Model Type not supported: {type(model)}')

    def deploy_sklearn(self, model, model_name=None):
        """Deploys a Scikit Learn model
        See `deploy_model()`
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError('Model is not a valid Scikit Learn estimator')

        model_class = type(model).__name__
        if not model_name:
            model_name = f'{model_class} {datetime.now().isoformat()}'

        # Serialize model
        content = io.BytesIO()
        pickle.dump(model, content)

        return self._create_model('sklearn',
                                  model_name=model_name,
                                  model_content=content.getvalue())

    def _create_model(self, type, model_name, model_content):
        response = self._api_call('/models',
                                  method='POST',
                                  json={
                                      'name': model_name,
                                      'type': type
                                  })
        upload_data = response['upload_data']
        print(f'Uploading model to Blazee...')
        upload_resp = requests.post(upload_data['url'],
                                    data=upload_data['fields'],
                                    files={'file': model_content})
        upload_resp.raise_for_status()

        print(f"Successfully deployed model {response['id']}")

        model = BlazeeModel(self, response)
        print(f"Deploying model... This will take a few moments")
        try:
            self._wait_for_depoyment(model)
        except TimeoutError:
            model.delete()
            raise RuntimeError("Error deploying model")
        return model

    def _api_call(self, path, method='GET', json=None):
        if json is None:
            data = None
        else:
            data = jsondumps(json, cls=NumpyEncoder)
        resp = requests.request(method=method,
                                url=f'{BLAZEE_HOST}{path}',
                                data=data,
                                headers={
                                    'X-Api-Key': self.api_key,
                                    'Content-Type': 'application/json'
                                })
        if resp.status_code >= 500:
            raise HTTPError(
                f'{resp.status_code} Internal Server Error: Please contact us at support@blazee.io')
        elif resp.status_code == 403:
            raise HTTPError(
                f'Invalid API Key. Get your API key on https://blazee.io')
        else:
            body = resp.json()
            if 'error' in body:
                error = body['error']
                error_msg = f"{resp.status_code} {body['error']['code']}: {body['error']['message']}"
                for details in error['details']:
                    error_msg += f'\n{details}'

                raise HTTPError(error_msg)
            else:
                return resp.json()

    def _wait_for_depoyment(self, model, wait=5, max_tries=30):
        for _ in range(max_tries):
            m = self.get_model(model.id)
            if m.uploaded:
                return
        raise TimeoutError('Deployment timeout')
