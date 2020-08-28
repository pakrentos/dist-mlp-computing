from flask import Flask, request
import json
from flask import request
from sql import *
from analyzer import *
from settings import *
from uuid import uuid4
from sqlite3 import connect
from datetime import datetime

conn = connect('users.db')
create_users(conn)

data = get_data()
print('Ready: ' + str(len(data['info']['Ready'])))
print('Processing: ' + str(len(data['info']['Processing'])))
print('Processed: ' + str(len(data['info']['Processed'])))
print(type(data['info']['Ready'][0]))

app = Flask(__name__)
if __name__ == '__main__':
    app.run(host='0.0.0.0')

def validate_secret_key(req_data):
    if req_data is None:
        return {'error': 'bad json'}, 403
    if req_data.get('secret_key', None) != SECRET_KEY:
        return {'error': 'bad secret key'}, 403
    else:
         return None

@app.route('/get_work_done', methods=['GET', 'POST'])
def get_work_done():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    return data

@app.route('/receive_job', methods=['GET', 'POST'])
def receive_job():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    req_data = request.get_json()
    if req_data['version'] != CODE_VERSION:
        return {'signal': 'update'}, 200
    try:
        ready_job = tuple(data['info']['Ready'].pop(0))
    except ValueError:
        print('We are done')
        return {'signal': 'done'}
    data['info']['Processing'].append(ready_job)
    sig = 'ok' if CODE_VERSION == req_data['version'] else 'update'
    temp_dict = {
        'subject': ready_job[0],
        'channels': ready_job[1],
        'group': GROUP,
        'hand': HAND,
        'preparing_type': PREPARING_TYPE,
        'clusters': CLUSTERS,
        'processing_type': PROCESSING_TYPE,
        'band': BAND,
        'signal': sig
        }
    if get_user(conn, req_data['UUID']) is None:
        insert_user(conn, req_data['UUID'], CODE_VERSION)
    update_first_time(conn, req_data['UUID'])
    print(f'New job sent. {temp_dict["subject"]} {temp_dict["channels"]}')
    print('Ready: ' + str(len(data['info']['Ready'])))
    print('Processing: ' + str(len(data['info']['Processing'])))
    print('Processed: ' + str(len(data['info']['Processed'])))
    return temp_dict

@app.route('/send_job', methods=['GET', 'POST'])
def send_job():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    req_data = request.get_json()
    sig = 'ok' if CODE_VERSION == req_data['version'] else 'update'
    # if (
    #     req_data['job']['group'] != GROUP or 
    #     req_data['job']['hand'] != HAND or 
    #     req_data['job']['preparing_type'] != PREPARING_TYPE or 
    #     req_data['job']['clusters'] != CLUSTERS or
    #     req_data['job']['band'] != BAND ):
    #     return {'signal': sig}, 200
    
    scores_len = [len(scores) for scores in req_data['data']['scores']]
    if min(scores_len) < CLUSTERS or len(scores_len) != 15:
        return  {'error': 'inconsistent data'}, 200
    
    data_label = (req_data['subject'], req_data['channels'])
    if data['data'].get(req_data['subject'], None) is None:
        data['data'][req_data['subject']] = {}

    if data_label in data['info']['Processed']:
        print('WARNING! Duplicate data! Calculating the mean...')
        for i in range(15):
            for j in range(CLUSTERS):
                old = data['data'][req_data['subject']][req_data['channels']][f'Trial {i}'][f'Cluster {j}']
                new = req_data['data']['scores'][i][j]
                data['data'][req_data['subject']][req_data['channels']][f'Trial {i}'][f'Cluster {j}'] = (old+new)/2
    else:
        data['data'][req_data['subject']][req_data['channels']] = {}
        for i in range(15):
            data['data'][req_data['subject']][req_data['channels']][f'Trial {i}'] = {}
            for j in range(CLUSTERS):
                data['data'][req_data['subject']][req_data['channels']][f'Trial {i}'][f'Cluster {j}'] = req_data['data']['scores'][i][j]
    try:
        data['info']['Processing'].remove(data_label)
    except ValueError:
        print("can't remove label from processing")
        pass
    if data_label in data['info']['Ready']:
        data['info']['Ready'].remove(data_label)
    data['info']['Processed'].append(data_label)
    file = open(PROCESSED_DATA_JSON_NAME, 'w')
    json.dump(data, file)
    file.close()
    print('Ready: ' + str(len(data['info']['Ready'])))
    print('Processing: ' + str(len(data['info']['Processing'])))
    print('Processed: ' + str(len(data['info']['Processed'])))
    update_last_time(conn, req_data['UUID'])
    return {'signal': sig}, 200

@app.route('/receive_update', methods=['GET', 'POST'])
def receive_update():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    req_data = request.get_json()
    file = open('slave.py', 'r')
    code = file.read()
    file.close
    ret_dict = JSON_CODE_VERSION
    ret_dict['code'] = code
    ret_dict['signal'] = 'ok'
    if req_data.get("UUID", None) is None:
        uuid = uuid4()
        insert_user(conn, uuid, CODE_VERSION)
        ret_dict['UUID'] = str(uuid)
    else:
        uuid = req_data['UUID']
        if get_user(conn, req_data['UUID']) is None:
            insert_user(conn, req_data['UUID'], CODE_VERSION)
        update_version(conn, uuid, CODE_VERSION)
    return ret_dict, 200

@app.route('/check_version', methods=['GET', 'POST'])
def check_version():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    return {'version': CODE_VERSION}, 200

@app.route('/error_report', methods=['GET', 'POST'])
def error_report():
    ret = validate_secret_key(request.get_json())
    if ret is not None:
        return ret
    req_data = request.get_json()
    print(f'Error on {req_data["UUID"]}, version {req_data["version"]}')
    file = open('logs.txt', 'a')
    file.write(f'{datetime.now()}: Error on {req_data["UUID"]}, version {req_data["version"]}')
    file.close()
    return {'signal': 'ok'}, 200
