import requests
import json
import sys
import subprocess
import os
from time import time
from tensorflow.keras import backend as K, layers
from tensorflow import keras
import numpy as np
from os.path import join
from scipy.signal import butter, lfilter
import pandas as pd
from operator import sub
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from toolz import curry
import warnings
warnings.filterwarnings('ignore')
SUBJECTS_DIR = 'Subjects'
import tensorflow as tf
import logging

### EEG Utils

channels = ['O1', 'Oz']
r_mapping = {'O2': 0, 'O1': 1, 'P4': 2, 'P3': 3, 'C4': 4, 'C3': 5, 'F4': 6,
             'F3': 7, 'Fp2': 8, 'Fp1': 9, 'T6': 10, 'T5': 11, 'T4': 12,
             'T3': 13, 'F8': 14, 'F7': 15, 'Oz': 16, 'Pz': 17, 'Cz': 18,
             'Fz': 19, 'Fpz': 20, 'FT7': 21, 'FC3': 22, 'Fcz': 23, 'FC4': 24,
             'FT8': 25, 'TP7': 26, 'CP3': 27, 'Cpz': 28, 'CP4': 29, 'TP8': 30}
def norm(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

def extract(data_stream):
    raw_data = data_stream.readlines()
    data = []
    for line in raw_data:
        temp = np.array([float(x) for x in line.split()])
        data.append(temp)
    return np.array(data)


def format_fname(_group, _hand, _subj, _tr_num):
    directory = SUBJECTS_DIR
    fname_pattern = f'{_group}_subject_{_subj}_{_hand}_tr_{_tr_num}.dat'
    return join(directory, fname_pattern)


def extract_all(group=None, hand=None, subject=None, trial=None):
    hands = ('lefthand', 'righthand')
    groups = ('OLD', 'YOUNG')
    subjects = np.arange(1, 11)
    trials = np.arange(1, 16)
    epochs = []
    group = groups if group not in groups else (group, )
    hand = hands if hand not in hands else (hand, )
    subject = subjects if subject not in subjects else (subject, )
    trial = trials if trial not in trials else (trial, )
    for g in group:
        for h in hand:
            for s in subject:
                for t in trial:
                    fin = open(format_fname(g, h, s, t))
                    temp = np.array(extract(fin))
                    epochs.append(temp)
                    fin.close()
    return np.array(epochs)


def select_channel(arr, ch):
    if ch in r_mapping.keys():
        ch = r_mapping[ch]
    elif ch not in r_mapping.values():
        raise ValueError('Wrong channel')
    mesh = [np.arange(arr.shape[i]) for i in range(len(arr.shape) - 2)] + [np.array([ch]), np.arange(arr.shape[-1])]
    ixgrid = np.ix_(*mesh)
    return np.squeeze(arr[ixgrid])

### Filtering

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

### Machine Learning Utils

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def R_squared(y, y_pred):
    residual = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y, y_pred)))
    total = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y, tf.math.reduce_mean(y))))
    r2 = tf.math.subtract(1.0, tf.math.divide(residual, total))
    return r2


def norm(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def normalize(arr):
    shape = arr.shape
    its = 1
    size = shape[-1]
    temp_arr = arr.flatten()
    for shape_i in shape:
        its *= shape_i
    its //= size
    ind = None
    for i in range(its - 1):
        temp_slice = temp_arr[i * size: (i + 1) * size]
        temp_arr[i * size: (i + 1) * size] = (temp_slice - np.min(temp_slice)) / (
                    np.max(temp_slice) - np.min(temp_slice))
        ind = i
    temp_slice = temp_arr[(ind + 1) * size:]
    temp_arr[(ind + 1) * size:] = (temp_slice - np.min(temp_slice)) / (np.max(temp_slice) - np.min(temp_slice))
    return temp_arr.reshape(*shape)


class EarlyStopDifference(keras.callbacks.Callback):
    # Custom callback that stops training if the difference between training and validation
    # loss function is more than delta for the past patience training epochs

    ### Parameters:
    ### min_delta: Integer, default=0; Minimal affordable difference between loss functions
    ### patience: Integer, default=0; Number of epochs it is tolerable to have difference greater than delta
    ### verbose: Integer, default=0; Prints output if it is 1

    def __init__(self, patience=0, min_delta=0.0, verbose=0):
        # Initializing parameters
        super(EarlyStopDifference, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Receiving loss function values
        mse = logs['loss']
        val_mse = logs['val_loss']
        # Comparing them to delta
        if self.verbose == 1:
            print("Epoch: ", epoch, "; Value: ", abs(mse - val_mse))
        if (abs(mse - val_mse) <= self.min_delta):
            # Resetting counter
            self.wait = 0
        else:
            # Incrementing counter
            self.wait += 1
            if self.wait >= self.patience:
                # Stopping the model if wait >= patience
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.flag = 1

    def on_train_end(self, logs=None):
        # Printing the epoch the model has stopped
        if (self.stopped_epoch > 0) and (self.verbose == 1):
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def generate_data(inp_set, out_set):
    """ Generates training and validation sets based on inputs and outputs

        input: numpy.array
        output: numpy.array
    """
    data = np.stack((inp_set, out_set), axis=-2)
    data = np.random.permutation(data)
    # Appear in order init_src, init_target, val_src, val_target
    return data[:data.shape[0] // 2, 0, :], data[:data.shape[0] // 2, 1, :], data[data.shape[0] // 2:, 0, :], data[
                                                                                                              data.shape[
                                                                                                                  0] // 2:,
                                                                                                              1, :]


def train_network(inputs, outputs, model, num_epochs=90000, verbose=False):
    init_src, init_trgt, val_src, val_trgt = generate_data(inputs, outputs)
    r2 = 0
    i = 1
    start = time()
    while model.flag != 0:
        if verbose:
            print('Model: ', i, 'Training start!')
        model.flag = 0
        r2 = model.fit(init_src, init_trgt,
                            epochs=num_epochs, batch_size=64, callbacks=my_callbacks,
                            validation_data=(val_src, val_trgt),
                            verbose=0).history['val_coeff_determination'][-1]
        elapsed = time() - start
        if elapsed > 30:
            start = time()
            del init_src
            del init_trgt
            del val_src,
            del val_trgt
            init_src, init_trgt, val_src, val_trgt = generate_data(inputs, outputs)
    del init_src
    del init_trgt
    del val_src,
    del val_trgt
    return r2

### Keras Model

class ResearchModel(keras.Sequential):
    def __init__(self):
        super().__init__()
        self.flag = 1

    def generate_weights(self, loc=0, scale=0.5):
        """
        weights: list of numpy arrays with weights of the model
        """
        result = []
        for w in self.get_weights():
            arr = np.random.normal(loc, scale, size=w.shape)
            arr = arr.astype(np.float32)
            result.append(arr)
        self.set_weights(result)
        self.flag = 1


def baseline_model(inputs=5, outputs=5):
    # Creates a keras neural network model
    model = ResearchModel()
    model.add(layers.Dense(inputs, activation='linear'))
    model.add(layers.Dense(30, kernel_initializer='random_normal', activation='elu'))
    model.add(layers.Dense(40, kernel_initializer='random_normal', activation='tanh'))
    model.add(layers.Dense(outputs, activation='linear'))
    # Компиляция модели
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.005), metrics=[coeff_determination])
    return model


my_callbacks = [
    EarlyStopDifference(patience=10, min_delta=0.01, verbose=0),
    #     keras.callbacks.EarlyStopping(patience=99999, monitor='val_coeff_determination', min_delta=0.001, restore_best_weights=True),
    keras.callbacks.EarlyStopping(patience=15, min_delta=0.0001, restore_best_weights=False),
]

### Phase Space Reconstruction

# параметры для альфа частот (8-12 гц): 6 lag, 5 dims
# для теты (4-8 гц): 11 lag, 5 dims
def reconstruct(x, lag, n_dims):
### Юзать ее
    x = _vector(x)

    if lag * (n_dims - 1) >= x.shape[0] // 2:
        raise ValueError('longest lag cannot be longer than half the length of x(t)')

    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()


def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=10, **cutoffs):
    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts


def _gfnn(x, lag, n_dims, **cutoffs):
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])


def _radius(x):
    return np.sqrt(((x - x.mean())**2).mean())


@curry
def _is_true_neighbor(
        x, attractor_radius, offset, indices, distance,
        relative_distance_cutoff=15,
        relative_radius_cutoff=2
):
    distance_increase = np.abs(sub(*x[indices + offset]))
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)


def _nearest_neighbors(y):
    distances, indices = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]


def ami(x, y=None, n_bins=10):
    x, y = _vector_pair(x, y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('timeseries must have the same length')
    return metrics.mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins=n_bins)[0])


def lagged_ami(x, min_lag=0, max_lag=None, lag_step=1, n_bins=10):
    if max_lag is None:
        max_lag = x.shape[0]//2
    lags = np.arange(min_lag, max_lag, lag_step)

    amis = [ami(reconstruct(x, lag, 2), n_bins=n_bins) for lag in lags]
    return lags, np.array(amis)


def _vector_pair(a, b):
    a = np.squeeze(a)
    if b is None:
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]
    return a, np.squeeze(b)


def _vector(x):
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x


def determine_coefs(x, min_lag=0, max_lag=20, min_dims=1, max_dims=10):
### Юзать ее
    lag_i, lag_d = lagged_ami(x, min_lag=min_lag, max_lag=max_lag, n_bins=10)
    lag = lag_i[np.argmin(lag_d)]
    dim_i, dim_d = global_false_nearest_neighbors(x, lag, max_dims=max_dims, min_dims=min_dims)
    ind = 0
    for i in range(len(dim_d)):
        if dim_d[i] == 0:
            ind = i
            break
    dim = dim_i[ind+1]
    return lag, dim

### Worker Class

class ComputeEeg:

    def __init__(self, subject=1, hand='righthand', group='YOUNG', channels=None, hc=12.0, lc=8.0, wn_num=10):
        if channels is None:
            channels = ['O1', 'O2']
        self.subject = subject
        self.channels = channels
        self.filtered_channels = self.filter_channels(channels, hc, lc, subject, group, hand)
        self.wn_num = wn_num

    def filter_channels(self, channels, hc, lc, subject, group, hand):
        ### Parameters:
        ### channels: array, String names of channels
        ### hc: float, high cut of frequency
        ### lc: float, low cut of frequency
        result = {}
        tests = extract_all(group=group, hand=hand, subject=subject)
        for ch in channels:
            result[ch] = {}
            inp_data = select_channel(tests, ch)
            result[ch]['data'] = []
            result[ch]['lag'] = []
            result[ch]['dim'] = []
            for i in range(0, 15):
                ch_data = butter_bandpass_filter(inp_data[i, :], lc, hc, 250)
                ch_data = ch_data[250 * 2 + 1:]
                ch_data = ch_data[:, np.newaxis]
                ch_lag, ch_dim = determine_coefs(ch_data, max_lag=10, max_dims=20)
                result[ch]['data'].append(ch_data)
                result[ch]['lag'].append(ch_lag)
                result[ch]['dim'].append(ch_dim)
        del tests
        return result

    def mlp_train(self):
        ### Trains neural network and returns array
        r2 = []
        hist = []
        dps = self.filtered_channels[self.channels[0]]['data'][0].shape[0] // self.wn_num
        ch1 = self.channels[0]
        ch2 = self.channels[1]
        for i in range(0, 15):
            lag = max(self.filtered_channels[ch1]['lag'][i], self.filtered_channels[ch2]['lag'][i])
            dim = max(self.filtered_channels[ch1]['dim'][i], self.filtered_channels[ch2]['dim'][i])
            inp = self.filtered_channels[ch1]['data'][i]
            out = self.filtered_channels[ch2]['data'][i]
            r = self.trial_train(inp, out, dps, lag, dim)
            r2.append(r)
        return r2

    def trial_train(self, inp, out, dps, lag, dim):
        r2 = []
        for i in range(0, self.wn_num):
            tf.keras.backend.clear_session()
            model = baseline_model(inputs=dim, outputs=dim)
            inputs = reconstruct(inp[i * dps:(i + 1) * dps, :], lag, dim)
            outputs = reconstruct(out[i * dps:(i + 1) * dps, :], lag, dim)
            inputs = norm(inputs)
            outputs = norm(outputs)
            r = train_network(inputs, outputs, model)
            r2.append(float(r))
            del model
            del inputs
            del outputs
        return r2

def execute_job(subject=1, hand='righthand', group='YOUNG', channels=None, hc=12.0, lc=8.0, wn_num=10):
    train_inst = ComputeEeg(subject, hand, group, channels, hc, lc, wn_num)
    r2 = train_inst.mlp_train()
    del train_inst
    return r2
mapping = {str(v):k for k,v in r_mapping.items()}

logging.basicConfig(filename='machine.log', level=logging.DEBUG, format='SLAVE: %(levelname)s: %(asctime)s: %(message)s')


# def execute_job(subject=1, hand='righthand', group='YOUNG', channels=None, hc=12.0, lc=8.0, wn_num=10):
#     r2 = np.random.normal(0.5, 0.2,size=(15, wn_num)).tolist()
#     return r2

def try_conn(uri, data):
    try:
        x = requests.post('http://18.234.193.208:5000/' + uri, json=data)
        try:
            return x.json()
        except ValueError:
            logging.warning('Server returned non-JSON answer')
            return None
    except requests.exceptions.ConnectionError:
        logging.warning('Server does not answer')
        return None

def get_self_version():
    code = open('slave.py', 'r', encoding='utf-8')
    _, version, uuid = code.readlines()[0].split()
    code.close()
    return uuid, version

def send_data_json():
    ret_dict = {'secret_key': "4C06037D-35D0-4693-BDFB-B36BDE89F725"}
    uuid, version = get_self_version()
    ret_dict['UUID'] = uuid
    ret_dict['version'] = version
    return ret_dict

def receive_job():
    answer = None
    while answer is None:
        answer = try_conn('receive_job', send_data_json())
    print(f'Received job {answer["subject"]} {answer["channels"]}')
    logging.info(f'Received job {answer["subject"]} {answer["channels"]}')
    answer['complete'] = False
    file = open('temp_job.json', 'w')
    json.dump(answer, file)
    file.close()
    if answer['signal'] == 'update' or answer['signal'] == 'done':
        try:
            os.remove('temp_job.json')
            os.remove('temp_data.json')
        except FileNotFoundError:
            pass
        exit()
    return answer

def send_job(send_job_data):
    send_job_data.update(send_data_json())
    send_job_data['complete'] = False
    file = open('temp_data.json', 'w')
    json.dump(send_job_data, file)
    file.close()
    answer = None
    while answer is None:
        answer = try_conn('send_job', send_job_data)
    logging.info(f'Sent job {send_job_data["subject"]} {send_job_data["channels"]}')
    send_job_data['complete'] = True
    file = open('temp_data.json', 'w')
    json.dump(send_job_data, file)
    file.close()
    if answer['signal'] == 'update':
        try:
            os.remove('temp_job.json')
            os.remove('temp_data.json')
        except FileNotFoundError:
            pass
        exit()
    return answer

def process_job(job):
    subject = int(job['subject'].split()[-1])
    ch1, ch2 = job['channels'].split()[-2:]
    channels = [mapping[ch1], mapping[ch2]]
    hand = job['hand']
    group = job['group']
    lc, hc = job['band']
    wn_num = job['clusters']
    scores = execute_job(subject=subject, hand=hand, group=group, channels=channels, hc=hc, lc=lc, wn_num=wn_num)
    prepare_to_send = {'job': job, 'subject': job['subject'], 'channels': job['channels'], 'data':{'scores':scores}}
    job['complete'] = True
    logging.info(f'Processed job {job["subject"]} {job["channels"]}')
    file = open('temp_job.json', 'w')
    json.dump(job, file)
    file.close()
    return prepare_to_send

def check_complete_last():
    try:
        file = open('temp_job.json', 'r')
        try:
            temp_data = json.load(file)
            if temp_data['complete'] == False:
                return temp_data
            else:
                return None
        except json.JSONDecodeError:
            return None
        file.close()
    except FileNotFoundError:
        return None

def check_sent_last():
    try:
        file = open('temp_data.json', 'r')
        try:
            temp_data = json.load(file)
            if temp_data['complete'] == False:
                return temp_data
            else:
                return None
        except json.JSONDecodeError:
            return None
        file.close()
    except FileNotFoundError:
        return None

def main():
    job = check_complete_last()
    if job is not None:
        prepare_to_send = process_job(job)
        send_job(prepare_to_send)
    else:
        job_data = check_sent_last()
        if job_data is not None:
            send_job(job_data)
    job = receive_job()
    prepare_to_send = process_job(job)
    send_job(prepare_to_send)

while True:
    main()