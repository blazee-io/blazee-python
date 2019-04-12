"""
Python Library for blazee.io
============================

Blazee is the easiest and fastest way to turn your Machine Learning
models and pipelines into a production ready prediction API.

At the moment, it supports Scikit Learn models

Usage:
    # Train your model like you usually do
    >>> from sklearn.linear_model import LogisticRegressionCV
    >>> clf = LogisticRegressionCV()
    >>> ...
    >>> clf.train(X)
    
    # Deploy your model on Blazee
    # Get your API Key on https://blazee.io
    >>> from blazee import Blazee
    >>> bz = Blazee(YOUR_API_KEY)
    >>> model = bz.deploy_model(clf)
    Uploading model to Blazee...
    Successfully deployed model bdea76f4-fa0f-4ef1-8bc5-f36978a4488e
    Deploying model... This will take a few moments
    
    # Predict a single sample
    >>> pred = model.predict(X[0])
    >>> pred.prediction
    1
    >>> pred.probas
    {0: 0.08, 1:0.91, 2: 0.01}

    # Or predict a batch
    >>> preds = model.batch_predict(X)
"""
__version__ = '0.1.4'

from .client import Client as Blazee
from .model import BlazeeModel
from .prediction import Prediction

__all__ = ["Blazee", "BlazeeModel", "Prediction", "__version__"]
