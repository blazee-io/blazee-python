
import io
from datetime import datetime

import joblib
import requests
from sklearn.base import BaseEstimator

from .model import Model

BLAZEE_HOST = 'https://apidev.blazee.io'


class Client:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('API Key must not be empty')
        self.api_key = api_key

    def deploy(self, model, model_name=None):
        if isinstance(model, BaseEstimator):
            self.deploy_sklearn(model, model_name)
        else:
            raise TypeError(f'Model Type not supported: {type(model)}')

    def deploy_sklearn(self, model, model_name=None):
        if not isinstance(model, BaseEstimator):
            raise TypeError('Model is not a valid Scikit Learn estimator')

        model_class = type(model).__name__
        if not model_name:
            model_name = f'{model_class} {datetime.now().isoformat()}'

        model_data = {
            "class": model_class,
            "params": model.get_params()
        }

        # Serialize model
        content = io.BytesIO()
        joblib.dump(model, content)

        model = self._create_model('sklearn',
                                   model_name=model_name,
                                   model_content=content.getvalue(),
                                   model_data=model_data)

    def _create_model(self, type, model_name, model_content, model_data):
        url = f'{BLAZEE_HOST}/v1/model'
        resp = requests.post(url,
                             headers={
                                 'X-Api-Key': self.api_key
                             },
                             json={
                                 'name': model_name,
                                 'type': type,
                                 'data': model_data
                             })

        if not resp:
            raise RuntimeError(f'Error creating model: {resp.text}')

        response = resp.json()
        upload_data = response['upload_data']
        print(f'Uploading model to Blazee...')
        upload_resp = requests.post(upload_data['url'],
                                    data=upload_data['fields'],
                                    files={'file': model_content})
        if not upload_resp:
            return RuntimeError(f'Error uploading model: {upload_resp.text}')

        print(f"Successfully deployed model {response['id']}")

        return Model(id=response['id'],
                     name=response['name'],
                     type=response['type'],
                     data=response['data'])
