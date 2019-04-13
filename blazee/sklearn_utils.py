import pickle

from blazee.keras_utils import get_keras_deps
from blazee.utils import (SerializedModel, add_file_deps,
                          get_files_dependencies, get_requirements)


def is_sklearn(model):
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return True
    except:
        pass

    # Check for Keras SK Wrapper
    try:
        from keras.wrappers.scikit_learn import BaseWrapper
        if isinstance(model, BaseWrapper):
            return True
    except:
        pass

    return False


def _get_estimator_dependencies(estimator):
    # Keras
    try:
        from keras.wrappers.scikit_learn import BaseWrapper
        if isinstance(estimator, BaseWrapper):
            return get_keras_deps()
    except:
        pass
    # XGBoost
    try:
        from xgboost.sklearn import XGBClassifier
        if isinstance(estimator, XGBClassifier):
            return ['xgboost']
    except:
        pass
    # LightGBM
    try:
        from lightgbm.sklearn import LGBMModel
        if isinstance(estimator, LGBMModel):
            return ['lightgbm']
    except:
        pass

    return []


def _get_model_metadata(model, include_files):
    deps = ['scikit-learn']
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator
    from sklearn.model_selection._search import BaseSearchCV

    # Import other depdendencies if needed
    if isinstance(model, Pipeline):
        for _, estimator in model.steps:
            deps += _get_estimator_dependencies(estimator)
    elif isinstance(model, BaseSearchCV):
        deps += _get_estimator_dependencies(model.estimator)
    elif isinstance(model, BaseEstimator):
        deps += _get_estimator_dependencies(model)
    else:
        raise ValueError(f"Model of type {type(model)} not supported")

    if include_files:
        deps += get_files_dependencies(include_files)

    return {
        'lib_versions': get_requirements(deps),
        'include_files': include_files
    }


def serialize_sklearn(model, include_files):
    # Check if model is fitted
    try:
        model.predict([0])
    except Exception as e:
        from sklearn.exceptions import NotFittedError
        if isinstance(e, NotFittedError):
            raise AttributeError("This model hasn't been trained yet")

    content = pickle.dumps(model)
    files = [('model.pickle', content)]

    add_file_deps(files, include_files)
    meta = _get_model_metadata(model, include_files)
    return SerializedModel('sklearn', meta, files)
