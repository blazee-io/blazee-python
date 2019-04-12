import os

from blazee.utils import (SerializedModel, add_file_deps,
                          get_files_dependencies, get_requirements)


def is_keras(model):
    try:
        from keras.models import Model
        return isinstance(model, Model)
    except:
        return False


def serialize_keras(model, include_files):
    # Unfortunately we cannot do that in memory yet, so use temp file
    tmp_file = 'tmp.h5'
    try:
        model.save(tmp_file)
        with open(tmp_file, 'rb') as f:
            weights = f.read()
    finally:
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    files = [('model.h5', weights)]

    add_file_deps(files, include_files)

    meta = _get_model_metadata(model, include_files)
    return SerializedModel('keras', meta, files)


def get_keras_deps():
    deps = ['keras']
    import keras
    backend = keras.backend.backend()
    if backend == 'tensorflow':
        deps.append('tensorflow')
    elif backend == 'theano':
        deps.append('theano')
    elif backend == 'cntk':
        deps.append('cntk')
    else:
        raise NotImplementedError(
            f'At the moment we only support tensorflow backend for Keras')
    return deps


def _get_model_metadata(model, include_files):
    deps = get_keras_deps()
    return {
        'lib_versions': get_requirements(deps),
        'include_files': include_files
    }
