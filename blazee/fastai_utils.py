import io
import os

from blazee.utils import (SerializedModel, add_file_deps,
                          get_files_dependencies, get_requirements)


def is_fastai(model):
    try:
        from fastai.basic_train import Learner
        return isinstance(model, Learner)
    except:
        return False


def _get_model_metadata(model, include_files):
    deps = ['fastai']
    return {
        'lib_versions': get_requirements(deps),
        'include_files': include_files
    }


def serialize_fastai(model, include_files):
    buffer = io.BytesIO()
    model.export(buffer)
    files = [('export.pkl', buffer.getvalue())]

    add_file_deps(files, include_files)

    meta = _get_model_metadata(model, include_files)

    return SerializedModel('fastai', meta, files)
