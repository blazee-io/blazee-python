import io
import json
import logging
import re
import zipfile
from collections import namedtuple

import numpy as np
import pkg_resources

SerializedModel = namedtuple('SerializedModel', ('type', 'metadata', 'files'))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        obj_type = type(obj)
        name = f'{obj_type.__module__}.{obj_type.__name__}'
        if name in ['numpy.ndarray', 'pandas.core.series.Series', 'pytorch.Tensor']:
            return obj.tolist()
        elif name == 'pandas.core.frame.DataFrame':
            return np.array(obj).tolist()

        return json.JSONEncoder.default(self, obj)


def pretty_size(num_bytes):
    if num_bytes < 1024:
        return f'{num_bytes} B'
    elif num_bytes < 1024 * 1024:
        return f'{num_bytes / 1024:1f} KB'
    else:
        return f'{num_bytes / 1024 / 1024:1f} MB'


def generate_zip(files):
    zipped = io.BytesIO()

    with zipfile.ZipFile(zipped, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name, content in files:
            zinfo = zipfile.ZipInfo(file_name)
            zinfo.external_attr = 0o644 << 16  # give read access
            zf.writestr(zinfo, content)

    return zipped.getvalue()


def get_requirements(packages):
    deps = {}
    for pkg in packages:
        preqs = get_installed_versions(pkg)
        deps = {**deps, **preqs}
    return deps


def get_installed_versions(package, skip=None):
    skip = skip or []
    if package in skip:
        return {}
    skip.append(package)
    dist = pkg_resources.get_distribution(package)
    requirements = {package: dist.version}
    for require in dist.requires():
        dep = require.name
        dep_reqs = get_installed_versions(dep, skip=skip)
        requirements = {**requirements, **dep_reqs}
    return requirements


def get_files_dependencies(file_names):
    deps = set([])
    for file_name in file_names:
        deps |= get_file_dependencies(file_name)
    return list(deps)


def get_file_dependencies(file_name):
    import_pattern = re.compile(r'import (.+)')
    from_pattern = re.compile(r'from (.+) import (.+)')
    reqs = set([])
    with open(file_name, 'r') as f:
        for l in f.readlines():
            imp_match = import_pattern.match(l)
            from_match = from_pattern.match(l)
            if imp_match:
                imp = imp_match[1]
                req = parse_import(imp)
                if req:
                    reqs.add(req)
            elif from_match:
                imp = from_match[1]
                req = parse_import(imp)
                if req:
                    reqs.add(req)
    return reqs


def parse_import(imp):
    if imp.startswith('.'):
        # Relative import
        # TODO: Check it's included in include_files
        return None
    try:
        pkg_name = imp.split('.')[0]
        pkg_resources.get_distribution(pkg_name)
        return pkg_name
    except pkg_resources.DistributionNotFound:
        # Relative import
        # TODO: Check it's included in include_files
        return None


def get_file_content(path):
    buffer = io .BytesIO()
    with open(path, 'rb') as f:
        buffer.write(f.read())
    return buffer.getvalue()


def add_file_deps(files, include_files):
    if include_files:
        for f in include_files:
            fname = f'deps/{f}'
            logging.info(f'Adding file {fname}')
            files.append((fname, get_file_content(f)))
