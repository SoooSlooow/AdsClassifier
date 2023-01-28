from src.parsing_utils import read_json_params, rewrite_json_params
import os

params = read_json_params()
query_id = params['query_id'] + 1
process_finished = params['process_finished']
rewrite_json_params(query_id=query_id)

while not process_finished:
    os.system('python -m get_data')
    params = read_json_params()
    process_finished = params['process_finished']
rewrite_json_params(process_finished=0)
