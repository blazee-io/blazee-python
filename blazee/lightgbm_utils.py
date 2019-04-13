import io
import os

from blazee.utils import (SerializedModel, add_file_deps,
                          get_files_dependencies, get_requirements)


def is_lightgbm(model):
    try:
        from lightgbm.basic import Booster
        return isinstance(model, Booster)
    except:
        return False


def _get_model_metadata(model, include_files):
    deps = ['lightgbm']
    return {
        'lib_versions': get_requirements(deps),
        'include_files': include_files
    }


def serialize_lightgbm(model, include_files):
    tmp_file = 'tmp.txt'
    try:
        model.save_model(tmp_file)
        with open(tmp_file, 'rb') as f:
            buffer = f.read()
    finally:
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    files = [('model.txt', buffer)]

    add_file_deps(files, include_files)

    meta = _get_model_metadata(model, include_files)

    return SerializedModel('lightgbm', meta, files)
