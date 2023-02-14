import json

PARAMS_PATH = 'params.json'


def read_json_params():
    with open(PARAMS_PATH, 'r') as params_file:
        params = json.load(params_file)
    return params


def rewrite_json_params(**new_params):
    with open(PARAMS_PATH, 'r+') as params_file:
        params = json.load(params_file)
        for new_param in new_params:
            params[new_param] = new_params[new_param]
        params_file.seek(0)
        json.dump(params, params_file)
        params_file.truncate()
