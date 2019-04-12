import io

from blazee.utils import (SerializedModel, add_file_deps,
                          get_files_dependencies, get_requirements)


def is_pytorch(model):
    try:
        from torch.nn import Module
        return isinstance(model, Module)
    except:
        return False


def _get_model_metadata(model, include_files):
    deps = ['torch', 'numpy']  # For some reason numpy is not a torch dep
    if include_files:
        deps += get_files_dependencies(include_files)

    return {
        'lib_versions': get_requirements(deps),
        'include_files': include_files
    }


def serialize_pytorch(model, include_files):
    import torch
    buffer = io.BytesIO()
    torch.save(model, buffer)

    files = [('model.pickle', buffer.getvalue())]

    if not include_files:
        raise ValueError(
            "With PyTorch, you must include the file where your model is defined (class extending torch.nn.Module)")

    add_file_deps(files, include_files)

    meta = _get_model_metadata(model, include_files)

    return SerializedModel('pytorch', meta, files)
