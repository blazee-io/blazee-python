"""
blazee.client
=============

This module provides a Client for interacting with the Blazee API.
"""
import logging
import os
import uuid
from datetime import datetime
from json import dumps as jsondumps

import requests
from requests.exceptions import HTTPError

from blazee.model import BlazeeModel, _serialize_model
from blazee.utils import NumpyEncoder

DEFAULT_BLAZEE_HOST = 'https://api.blazee.io/v1'


class Client:
    """Client for interacting with the Blazee API.

    Parameters
    ----------
    api_key : string
        Your Blazee API key. Get one at https://blazee.io
        This can also be set through the BLAZEE_API_KEY environment variable

    host: string
        The base URL of the Blazee API. Defaults to
        Blazee production API.
        This can also be set through the BLAZEE_HOST environment variable
    """

    def __init__(self, api_key: str = None, host: str = None):
        if not host:
            host = os.environ.get('BLAZEE_HOST')
            if not host:
                host = DEFAULT_BLAZEE_HOST
        if not api_key:
            api_key = os.environ.get('BLAZEE_API_KEY')
            if not api_key:
                raise ValueError('API Key must not be empty')

        self.api_key = api_key
        self.host = host

    def all_models(self):
        """Returns a list of all your deployed Blazee models.

        Returns
        -------
        models: list of `blazee.model.BlazeeModel`
        """
        resp = self._api_call('/models')

        return [BlazeeModel(self, m) for m in resp]

    def get_model(self, model_id):
        """Returns the Blazee model with the given ID.

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

    def deploy_model(self, model, model_name=None, include_files=None):
        """Deploys a trained ML model on Blazee
        At the moment we support Scikit Learn, Keras and PyTorch models.
        Looking for another framework? Reach out at support@blazee.io

        Parameters
        ----------
        model: one of `sklearn.base.BaseEstimator`, `keras.models.Model`, `torch.nn.Module`
            The model to deploy
        model_name: string
            A custom name for the Blazee model
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
        serialized_model = _serialize_model(model, include_files=include_files)

        if not model_name:
            model_name = f'{type(model).__name__} {datetime.now().isoformat()}'

        model = self._create_model('sklearn', model_name=model_name)
        try:
            version = model._upload_version(serialized_model)
        except Exception as e:
            logging.info('Something wrong happened, deleting model...')
            model.delete()
            raise e

        return version.model

    def _create_model(self, type, model_name):
        response = self._api_call('/models',
                                  method='POST',
                                  json={
                                      'name': model_name
                                  })

        return BlazeeModel(self, response)

    def _api_call(self, path, method='GET', json=None):
        if json is None:
            data = None
        else:
            data = jsondumps(json, cls=NumpyEncoder)
        resp = requests.request(method=method,
                                url=f'{self.host}{path}',
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
