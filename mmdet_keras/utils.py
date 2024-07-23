import ast
from importlib import import_module
import os.path as osp
import platform
import re
import shutil
import sys
import tempfile
from addict import Dict


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def _substitute_predefined_vars(filename, temp_config_name):
    file_dirname = osp.dirname(filename)
    file_basename = osp.basename(filename)
    file_basename_no_extension = osp.splitext(file_basename)[0]
    file_extname = osp.splitext(filename)[1]
    support_templates = dict(
        fileDirname=file_dirname,
        fileBasename=file_basename,
        fileBasenameNoExtension=file_basename_no_extension,
        fileExtname=file_extname)
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        config_file = f.read()
    for key, value in support_templates.items():
        regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
        value = value.replace('\\', '/')
        config_file = re.sub(regexp, value, config_file)
    with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
        tmp_config_file.write(config_file)


def _validate_py_syntax(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                            f'file {filename}: {e}')


def _file2dict(filename, use_predefined_variables=True):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        # Substitute predefined variables
        if use_predefined_variables:
            _substitute_predefined_vars(filename, temp_config_file.name)
        else:
            shutil.copyfile(filename, temp_config_file.name)
        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        _validate_py_syntax(filename)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        # delete imported module
        del sys.modules[temp_module_name]

        # close temp file
        temp_config_file.close()

    return Dict(cfg_dict)


def make_layer(layer_object, layers_dict=None):
    if layers_dict is None:
        return layer_object

    layer_name = layer_object.name
    if layer_name in layers_dict:
        return layers_dict[layer_name]

    layers_dict[layer_name] = layer_object
    return layer_object
