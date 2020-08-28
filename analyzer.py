import json
from settings import *

def _ready():
    ready_arr = []
    for sub_n in range(1, 11):
        for ch in channels_overall_arr:
            ready_arr.append((f'Subject {sub_n}', f'Channels {ch[0]} {ch[1]}'))
    return ready_arr


def get_processed(data):
    ready = _ready()
    new_ready = []

    processed = []
    inds = [True for i in range(len(ready))]
    for i in range(len(ready)):
        if data.get(ready[i][0], None) is None:
            continue
        elif data[ready[i][0]].get(ready[i][1], None) is None:
            continue
        else:
            temp = data[ready[i][0]][ready[i][1]]
            num_of_data = [len(v) for v in temp.values()]
            if len(num_of_data) != 15 or min(num_of_data) < 10:
                continue
            else:
                processed.append(ready[i])
                inds[i] = False

    for i in range(len(ready)):
        if inds[i]:
            new_ready.append(ready[i])
    
    return new_ready, processed

def get_data():
    try:
        file = open(PROCESSED_DATA_JSON_NAME, 'r')
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {}
            data['info'] = {
                'Group': GROUP,
                'Hand': HAND,
                'Preparing': PREPARING_TYPE,
                'Processing method': PROCESSING_TYPE,
                'Band': BAND_NAME,
                'Number of clusters': CLUSTERS
            }
            data['data'] = {}
        file.close()
    except FileNotFoundError:
        data = {}
        data['info'] = {
            'Group': GROUP,
            'Hand': HAND,
            'Preparing': PREPARING_TYPE,
            'Processing method': PROCESSING_TYPE,
            'Band': BAND_NAME,
            'Number of clusters': CLUSTERS
        }
        data['data'] = {}

    if data['info'].get('Ready', None) is None or data['info'].get('Processed', None) is None or data['info'].get('Processing', None) is None:
        ready, processed = get_processed(data['data'])

    if data['info'].get('Ready', None) is None:
        data['info']['Ready'] = ready
    
    if data['info'].get('Processed', None) is None:
        data['info']['Processed'] = processed

    if data['info'].get('Processing', None) is None:
        data['info']['Processing'] = []
    for i in range(len(data['info']['Ready'])):
        data['info']['Ready'][i] = tuple(data['info']['Ready'][i])
    for i in range(len(data['info']['Processing'])):
        data['info']['Processing'][i] = tuple(data['info']['Processing'][i])
    for i in range(len(data['info']['Processed'])):
        data['info']['Processed'][i] = tuple(data['info']['Processed'][i])
    ready_set = set(data['info']['Ready'])
    processing_set = set(data['info']['Processing'])
    processed_set = set(data['info']['Processed'])
    intersection_r_p = ready_set & processing_set
    intersection_p_p = processing_set & processed_set
    data['info']['Ready'] = list(ready_set ^ intersection_r_p)
    data['info']['Processing'] = list(processing_set ^ intersection_p_p)
    return data