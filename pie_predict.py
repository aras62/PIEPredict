"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import os
import time
import pickle
import numpy as np

from keras.layers import Input, RepeatVector, Dense, Permute
from keras.layers import Concatenate, Multiply, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras import regularizers

class PIEPredict(object):
    """
    An encoder decoder model for pedestrian trajectory prediction

    Attributes:
       _num_hidden_units: Number of LSTM hidden units
       _regularizer_value: The value of L2 regularizer for training
       _regularizer: Training regularizer set as L2
       _activation: LSTM actications
       _embed_size: The size embedding unit applied to the representation produced by encoder
       _embed_dropout: the dropout of embedding unit

    Model attributes: The following attributes will be set during training depending on the training data
       _observe_length: Observation duration in frames (number of time steps of the encoder)
       _predict_length: Prediciton duration in frames (number of time steps of the decoder)
       _encoder_feature_size: The number of data points entering the encoder, e.g. for 'bounding boxes' the size is 4 (x1 y1 x2 y2)
       _decoder_feature_size: The number of data points entering the decoder, e.g. for 'speed' the size is 1
       _prediction_size: The size of the decoder output after dense layer, e.g. for 'bounding boxes' the size is 4 (x1 y1 x2 y2)

    Methods:
        get_tracks: Generates trajectory tracks by sampling from pedestrian sequences
        get_data_helper: Create training data by combining sequences with different modalities
        get_data: Generates training and testing data
        log_configs: Writes model and training configurations to a file
        train: Trains the model
        test: Tests the model
        pie_encdec: Generates the network model
        create_lstm_model: A helper function for creating an LSTM unit
        attention_temporal: Temporal attention custom layer
        attention_element: Elementwise attention custom layer
    """
    def __init__(self,
                 num_hidden_units=256,
                 regularizer_val=0.0001,
                 activation='softsign',
                 embed_size=64,
                 embed_dropout=0):

        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)

        self._activation = activation
        self._embed_size = embed_size
        self._embed_dropout = embed_dropout

        # model parameters
        self._observe_length = 15
        self._predict_length = 15

        self._encoder_feature_size = 4
        self._decoder_feature_size = 4

        self._prediction_size = 4

    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']

        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in
                               range(0, len(track) - seq_length + 1, overlap_stride)])
            d[k] = tracks

        #  Normalize tracks by subtracting bbox/center at first time step from the rest
        if normalize:
            if 'bbox' in data_types:
                for i in range(len(d['bbox'])):
                    d['bbox'][i] = np.subtract(d['bbox'][i][1:], d['bbox'][i][0]).tolist()
            if 'center' in data_types:
                for i in range(len(d['center'])):
                    d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
            #  Adjusting the length of other data types
            for k in d.keys():
                if k != 'bbox' and k != 'center':
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][1:]

        return d

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        else:
            return d[0]

    def get_data(self, data, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """
        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': 15,
            'predict_length': 45,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }

        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])

        if opts['normalize_bbox']:
            observe_length -= 1

        obs_slices = {}
        pred_slices = {}

        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:observe_length] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length:] for d in data_tracks[k]])

        # Generate observation data input to encoder
        enc_input = self.get_data_helper(obs_slices, opts['enc_input_type'])

        # Generate data for prediction decoder
        dec_input = self.get_data_helper(pred_slices, opts['dec_input_type'])
        pred_target = self.get_data_helper(pred_slices, opts['prediction_type'])

        if not len(dec_input) > 0:
            dec_input = np.zeros(shape=pred_target.shape)

        return {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'enc_input': enc_input,
                'dec_input': dec_input,
                'pred_target': pred_target,
                'model_opts': opts}

    def get_path(self,
                 file_name='',
                 save_folder='models',
                 dataset='pie',
                 model_type='trajectory',
                 save_root_folder='data/'):
        """
        A path generator method for saving model and config data. It create directories if needed.
        :param file_name: The actual save file name , e.g. 'model.h5'
        :param save_folder: The name of folder containing the saved files
        :param dataset: The name of the dataset used
        :param save_root_folder: The root folder
        :return: The full path for the model name and the path to the final folder
        """
        save_path = os.path.join(save_root_folder, dataset, model_type, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path

    def log_configs(self, config_path, batch_size, epochs,
                    lr, loss, learning_scheduler, opts):
        """
        Logs the parameters of the model and training
        :param config_path: The path to save the file
        :param batch_size: Batch size of training
        :param epochs: Number of epochs for training
        :param lr: Learning rate of training
        :param loss: Type of loss function
        :param learning_scheduler: Whether learning scheduler was used
        :param opts: Model options (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            fid.write("####### Model options #######\n")
            for k in opts:
                fid.write("%s: %s\n" % (k, str(opts[k])))

            fid.write("\n####### Network config #######\n")
            fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
            fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))
            fid.write("%s: %s\n" % ('activation', str(self._activation)))
            fid.write("%s: %s\n" % ('embed_size', str(self._embed_size)))
            fid.write("%s: %s\n" % ('embed_dropout', str(self._embed_dropout)))

            fid.write("%s: %s\n" % ('observe_length', str(self._observe_length)))
            fid.write("%s: %s\n" % ('predict_length ', str(self._predict_length)))
            fid.write("%s: %s\n" % ('encoder_feature_size', str(self._encoder_feature_size)))
            fid.write("%s: %s\n" % ('decoder_feature_size', str(self._decoder_feature_size)))
            fid.write("%s: %s\n" % ('prediction_size', str(self._prediction_size)))

            fid.write("\n####### Training config #######\n")
            fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
            fid.write("%s: %s\n" % ('epochs', str(epochs)))
            fid.write("%s: %s\n" % ('lr', str(lr)))
            fid.write("%s: %s\n" % ('loss', str(loss)))
            fid.write("%s: %s\n" % ('learning_scheduler', str(learning_scheduler)))

        print('Wrote configs to {}'.format(config_path))

    def train(self, data_train, data_val,
              batch_size=64,
              epochs=60,
              lr=0.001,
              loss='mse',
              learning_scheduler=True,
              **model_opts):
        """
        Training method for the model
        :param data_train: Training data
        :param data_val: Validation data
        :param batch_size: Batch size of training
        :param epochs: Number of epochs for training
        :param lr: Learning rate of training
        :param loss: Type of loss function
        :param learning_scheduler: Whether learning scheduler was used
        :param model_opts: Data generation parameters (see get_data)
        :return: The path to where the final model is saved
        """
        optimizer = RMSprop(lr=lr)

        train_data = self.get_data(data_train, **model_opts)
        val_data = self.get_data(data_val, **model_opts)

        print("Number of samples:\n Train: %d \n Val: %d \n"
              % (train_data['enc_input'].shape[0], val_data['enc_input'].shape[0]))

        self._observe_length = train_data['enc_input'].shape[1]
        self._predict_length = train_data['pred_target'].shape[1]

        self._encoder_feature_size = train_data['enc_input'].shape[2]
        self._decoder_feature_size = train_data['dec_input'].shape[2]

        # Set the output sizes
        self._prediction_size = train_data['pred_target'].shape[2]

        # Set path names for saving configs and model
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        if 'bbox' in model_opts['prediction_type']:
            model_type = 'trajectory'
        else:
            model_type = 'speed'
        print(model_type)

        model_path, _ = self.get_path(save_folder=model_folder_name,
                                      model_type=model_type,
                                      file_name='model.h5')

        # Save data parameters
        opts_path, _ = self.get_path(save_folder=model_folder_name,
                                     model_type=model_type,
                                     file_name='model_opts.pkl')


        with open(opts_path, 'wb') as fid:
            pickle.dump(train_data['model_opts'], fid,
                        pickle.HIGHEST_PROTOCOL)

        # save training and model parameters
        config_path, _ = self.get_path(save_folder=model_folder_name,
                                       model_type=model_type,
                                       file_name='configs.txt')
        self.log_configs(config_path, batch_size, epochs,
                         lr, loss, learning_scheduler,
                         train_data['model_opts'])

        pie_model = self.pie_encdec()

        # Generate training data
        train_data = ([train_data['enc_input'],
                       train_data['dec_input']],
                      train_data['pred_target'])

        val_data = ([val_data['enc_input'],
                     val_data['dec_input']],
                    val_data['pred_target'])

        pie_model.compile(loss=loss, optimizer=optimizer)

        print("##############################################")
        print(" Training for predicting sequences of size %d" % self._predict_length)
        print("##############################################")

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss')
        call_backs = [checkpoint]

        #  Setting up learning schedulers
        if learning_scheduler:
            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=1.0, patience=10,
                                       verbose=1)
            plateau_sch = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2, patience=5,
                                            min_lr=1e-07, verbose=1)
            call_backs.extend([early_stop, plateau_sch])

        history = pie_model.fit(x=train_data[0], y=train_data[1],
                                batch_size=batch_size, epochs=epochs,
                                validation_data=val_data, verbose=1,
                                callbacks=call_backs)

        print('Train model is saved to {}'.format(model_path))

        history_path, saved_files_path = self.get_path(save_folder=model_folder_name,
                                                       model_type=model_type,
                                                       file_name='history.pkl')

        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        Testing method for the model
        :param data_test: Testing data
        :param model_path: The path to where the model to be tested is saved
        :return: Mean squared error (MSE) of the prediction
        """
        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()

        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_data = self.get_data(data_test, **model_opts)
        test_obs_data = [test_data['enc_input'], test_data['dec_input']]
        test_target_data = test_data['pred_target']

        test_results = test_model.predict(test_obs_data, batch_size=2048, verbose=1)

        perf = {}
        #  Performance on bounding boxes
        performance = np.square(test_target_data - test_results)
        perf['mse'] = performance.mean(axis=None)
        perf['mse_last'] = performance[:, -1, :].mean(axis=None)

        # print("MSE  %f" % perf['mse'])
        # print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
        #       % (perf['mse-15'], perf['mse-30'], perf['mse']))
        # print("MSE last %f" % perf['mse_last'])

        if model_opts['prediction_type'][0] == 'bbox':
            #  Performance on centers (displacement)
            model_opts['normalize_bbox'] = False
            test_data = self.get_data(data_test, **model_opts)
            test_obs_data_org = [test_data['enc_input'], test_data['dec_input']]
            test_target_data_org = test_data['pred_target']

            results_org = test_results + np.expand_dims(test_obs_data_org[0][:, 0, 0:4], axis=1)

            #  Performance measures for centers
            res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            for b in range(test_results.shape[0]):
                for s in range(test_results.shape[1]):
                    centers[b, s, 0] = (test_target_data_org[b, s, 2] + test_target_data_org[b, s, 0]) / 2
                    centers[b, s, 1] = (test_target_data_org[b, s, 3] + test_target_data_org[b, s, 1]) / 2
                    res_centers[b, s, 0] = (results_org[b, s, 2] + results_org[b, s, 0]) / 2
                    res_centers[b, s, 1] = (results_org[b, s, 3] + results_org[b, s, 1]) / 2

            c_performance = np.square(centers - res_centers)
            perf['center_mse'] = c_performance.mean(axis=None)
            perf['center_mse_last'] = c_performance[:, -1, :].mean(axis=None)

            # # print("Center MSE  %f" % perf['center_mse'])
            # print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f"
            #       % (perf['c-mse-15'], perf['c-mse-30'], perf['center_mse']))
            # print("Center MSE last %f" % perf['center_mse_last'])

        save_results_path = os.path.join(model_path,
                                         '{:.2f}.pkl'.format(perf['mse']))
        save_performance_path = os.path.join(model_path,
                                         '{:.2f}.txt'.format(perf['mse']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': data_test['pred_image'],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}

            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        return perf

    def test_final(self, data_test, traj_model_path='', intent_model_path='', speed_model_path=''):

        intent_path = os.path.join(intent_model_path, 'ped_intents.pkl')
        with open(intent_path, 'rb') as fid:
            try:
                intent = pickle.load(fid)
            except:
                intent = pickle.load(fid, encoding='bytes')

        model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,  # 0.8 for jaad, 0.5 pie
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': [],
                       'prediction_type': ['bbox']}

        box_data = self.get_data(data_test, **model_opts)

        intent_dic = {}
        for pid, img, r in zip(intent['ped_id'], intent['images'], intent['results']):
            img_name = img[0].split('/')[-1].split('.')[0]
            p_id = pid[0][0]
            if p_id in intent_dic.keys():
                intent_dic[p_id][img_name] = r
            else:
                intent_dic[p_id] = {img_name: r}

        int_data = np.zeros(shape=(box_data['pred_target'].shape[0], box_data['pred_target'].shape[1], 1))
        obs_pids = box_data['obs_pid']
        obs_images = box_data['obs_image']

        intent_list = []
        last_ped = ''
        for i in range(len(obs_pids)):
            pid = obs_pids[i][0][0]
            if pid != last_ped:
                intent_list = []
                last_ped = pid

            # check if the current id exists in intention results
            if pid in intent_dic:
                img_name = obs_images[i][0].split('/')[-1].split('.')[0]
                # if id exists we check if there is a value for the given sequence
                if img_name in intent_dic[pid]:
                    intent_result = intent_dic[pid][img_name]
                    intent_list.append(intent_result)
                    int_data[i] = np.array([intent_result] * box_data['pred_target'].shape[1])
                else:
                    # if there is no value we check whether previously not value was observed
                    if intent_list == []:
                        int_data[i] = np.array([[0.5]] * box_data['pred_target'].shape[1])
                    else:
                        # if there is no value we check and list is not empty, we use average results
                        int_avg = np.mean(np.array(intent_list))
                        int_data[i] = np.array([[int_avg]] * box_data['pred_target'].shape[1])
            else:
                # if the id does not exist just populate with 0.5
                int_data[i] = np.array([[0.5]] * box_data['pred_target'].shape[1])

        #speed_path = '/home/aras/PycharmProjects/release_code/test/speed_models/pie_speed'
        speed_model = load_model(os.path.join(speed_model_path, 'model.h5'))

        #bis_path = '/home/aras/PycharmProjects/release_code/test/box_intent_speed'
        box_intent_speed_model = load_model(os.path.join(traj_model_path, 'model.h5'))

        ################## run speed model ####################
        model_opts['enc_input_type'] = ['obd_speed']
        model_opts['prediction_type'] = ['obd_speed']
        speed_data = self.get_data(data_test, **model_opts)

        _speed_data = [speed_data['enc_input'], speed_data['dec_input']]
        speed_results = speed_model.predict(_speed_data,
                                            batch_size=2056,
                                            verbose=1)

        # speed intent
        int_speed = np.concatenate([int_data, speed_results], axis=2)
        test_results = box_intent_speed_model.predict([box_data['enc_input'], int_speed],
                                                      batch_size=2056, verbose=1)

        # Performance measures for bounding boxes
        perf = {}
        performance = np.square(test_results - box_data['pred_target'])
        perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
        perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)  # 15:30
        perf['mse-45'] = performance.mean(axis=None)
        perf['mse-last'] = performance[:, -1, :].mean(axis=None)

        # print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
        #       % (perf['mse-15'], perf['mse-30'], perf['mse-45']))
        # print("mse-last %.2f\n" % (perf['mse-last']))

        #  Performance on centers (displacement)
        model_opts['normalize_bbox'] = False
        model_opts['enc_input_type'] = ['bbox']
        model_opts['prediction_type'] = ['bbox']
        test_data = self.get_data(data_test, **model_opts)
        test_obs_data_org = [test_data['enc_input'], test_data['dec_input']]
        test_target_data_org = test_data['pred_target']

        results_org = test_results + np.expand_dims(test_obs_data_org[0][:, 0, 0:4], axis=1)

        #  Performance measures for centers
        res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
        centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
        for b in range(test_results.shape[0]):
            for s in range(test_results.shape[1]):
                centers[b, s, 0] = (test_target_data_org[b, s, 2] + test_target_data_org[b, s, 0]) / 2
                centers[b, s, 1] = (test_target_data_org[b, s, 3] + test_target_data_org[b, s, 1]) / 2
                res_centers[b, s, 0] = (results_org[b, s, 2] + results_org[b, s, 0]) / 2
                res_centers[b, s, 1] = (results_org[b, s, 3] + results_org[b, s, 1]) / 2

        c_performance = np.square(centers - res_centers)
        perf['c-mse-15'] = c_performance[:, 0:15, :].mean(axis=None)
        perf['c-mse-30'] = c_performance[:, 0:30, :].mean(axis=None)  # 0:30
        perf['c-mse-45'] = c_performance.mean(axis=None)
        perf['c-mse-last'] = c_performance[:, -1, :].mean(axis=None)

        # print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f" \
        #       % (perf['c-mse-15'], perf['c-mse-30'], perf['c-mse-45']))
        # print("c-mse-last: %.2f\n" % (perf['c-mse-last']))

        save_results_path = os.path.join(traj_model_path,
                                         '{:.2f}.pkl'.format(perf['mse-45']))
        save_performance_path = os.path.join(traj_model_path,
                                             '{:.2f}.txt'.format(perf['mse-45']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': box_data['pred_image'],
                           'results': test_results,
                           'gt': box_data['pred_target'],
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': test_results,
                           'gt': box_data['pred_target'],
                           'performance': perf}
            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)
        return perf  # performance

    def pie_encdec(self):
        """
        Generates the encoder decoder method
        :return: An instance of the network model
        """

        # Generate input data. the shapes is (sequence_lenght,length of flattened features)
        _encoder_input = Input(shape=(self._observe_length, self._encoder_feature_size),
                               name='encoder_input')

        # Temporal attention module
        _attention_net = self.attention_temporal(_encoder_input, self._observe_length)

        # Generate Encoder LSTM Unit
        encoder_model = self.create_lstm_model(name='encoder_network')
        _encoder_outputs_states = encoder_model(_attention_net)
        _encoder_states = _encoder_outputs_states[1:]

        # Generate Decoder LSTM unit
        decoder_model = self.create_lstm_model(name='decoder_network', r_state=False)
        _hidden_input = RepeatVector(self._predict_length)(_encoder_states[0])
        _decoder_input = Input(shape=(self._predict_length, self._decoder_feature_size),
                               name='pred_decoder_input')

        # Embedding unit on the output of Encoder
        _embedded_hidden_input = Dense(self._embed_size, activation='relu')(_hidden_input)
        _embedded_hidden_input = Dropout(self._embed_dropout,
                                         name='dropout_dec_input')(_embedded_hidden_input)

        decoder_concat_inputs = Concatenate(axis=2)([_embedded_hidden_input, _decoder_input])

        # Self attention unit
        att_input_dim = self._embed_size + self._decoder_feature_size
        decoder_concat_inputs = self.attention_element(decoder_concat_inputs, att_input_dim)

        # Initialize the decoder with encoder states
        decoder_output = decoder_model(decoder_concat_inputs,
                                       initial_state=_encoder_states)
        decoder_output = Dense(self._prediction_size,
                               activation='linear',
                               name='decoder_dense')(decoder_output)

        net_model = Model(inputs=[_encoder_input, _decoder_input],
                          outputs=decoder_output)
        net_model.summary()

        return net_model

    def create_lstm_model(self, name='lstm', r_state=True, r_sequence=True):
        """
        A Helper function that generates an instance of LSTM
        :param name: Name of the layer
        :param r_state: Whether to return states
        :param r_sequence: Whether to return sequences
        :return: An LSTM instance
        """

        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    activity_regularizer=None,
                    activation=self._activation,
                    name=name)

    # Custom layers
    def attention_temporal(self, input_data, sequence_length):
        """
        A temporal attention layer
        :param input_data: Network input
        :param sequence_length: Length of the input sequence
        :return: The output of attention layer
        """
        a = Permute((2, 1))(input_data)
        a = Dense(sequence_length, activation='sigmoid')(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([input_data, a_probs])
        return output_attention_mul

    def attention_element(self, input_data, input_dim):
        """
        A self-attention unit
        :param input_data: Network input
        :param input_dim: The feature dimension of the input
        :return: The output of the attention network
        """
        input_data_probs = Dense(input_dim, activation='sigmoid')(input_data)  # sigmoid
        output_attention_mul = Multiply()([input_data, input_data_probs])  # name='att_mul'
        return output_attention_mul
