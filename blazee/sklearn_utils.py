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


def _get_model_metadata(model, include_files):
    deps = ['scikit-learn']
    try:
        from keras.wrappers.scikit_learn import BaseWrapper
        if isinstance(model, BaseWrapper):
            deps += get_keras_deps()
    except:
        pass
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
