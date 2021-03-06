# Python Library for Blazee

## Intro

Blazee is the easiest and fastest way to turn your Machine Learning
models and pipelines into a production ready prediction API.

It allows you to deploy trained models straight from a Jupyter Notebook
or any other model training environment, and access them live from anywhere
using the Blazee HTTP API.

This library can also be used

## Supported Frameworks

At the moment, we support the following frameworks:

- Scikit Learn (Supervised learning models and pipeline)
- [Keras](https://keras.io/)
- [PyTorch](https://pytorch.org/)
- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/Microsoft/LightGBM)
- [Fast.ai](https://www.fast.ai/)

Coming soon:

- [H2O](https://www.h2o.ai/)
- [Tensorflow](https://www.tensorflow.org/)

## Installation

Install from pip

```shell
pip install blazee
```

Sign up and get an API Key from https://blazee.io

## Usage

```python
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

# Deploy another version of the model
>>> clf2 = SGDClassifier()
>>> ...
>>> clf2.train(X)
>>> model.update(clf2)
```

## Support

Contact us at support@blazee.io or open a Github Issue for any question or bug report.
