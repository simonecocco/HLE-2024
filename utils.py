from os.path import join, exists
from os import makedirs

def check_dir(path):
    if not exists(path):
        makedirs(path)


def get_data_dir(dataset_name):
    path = join('data', dataset_name)
    check_dir(path)
    return path

def get_template_dir():
    path = 'data/templates'
    check_dir(path)
    return path

def get_tokenizer_dir():
    path = 'tokenizers'
    check_dir(path)
    return path

def get_model_tokenizer_dir(model_name):
    path = join(get_tokenizer_dir(), model_name)
    check_dir(path)
    return path

def get_raw_paths_dir(dataset_name):
    path = join(get_data_dir(dataset_name), 'raw_paths')
    check_dir(path)
    return path

def get_filled_templates_path(dataset_name):
    path = join(get_data_dir(dataset_name), 'filled_templates')
    check_dir(path)
    return path

def get_tokenizer_dataset(dataset_name):
    path = join(get_tokenizer_dir(), dataset_name)
    check_dir(path)
    return path

def get_tokenized_dataset(dataset_name):
    path = join(get_data_dir(dataset_name), 'tokenized')
    check_dir(path)
    return path
