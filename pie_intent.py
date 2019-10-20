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

import numpy as np
import os
import pickle
import time

from keras import backend as K
from keras import regularizers
from keras.applications import vgg16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Concatenate
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils import *

#from utilities.jaad_eval import *
#from utilities.jaad_utilities import *
#from utilities.train_utilities import *

K.set_image_dim_ordering('tf')


class PIEIntent(object):
    """
    A convLSTM encoder decoder model for predicting pedestrian intention
    Attributes:
        _num_hidden_units: Number of LSTM hidden units
        _reg_value: the value of L2 regularizer for training
        _kernel_regularizer: Training regularizer set as L2
        _recurrent_regularizer: Training regularizer set as L2
        _activation: LSTM activations
        _lstm_dropout: input dropout
        _lstm_recurrent_dropout: recurrent dropout
        _convlstm_num_filters: number of filters in convLSTM
        _convlstm_kernel_size: kernel size in convLSTM

    Model attributes: set during training depending on the data
        _encoder_input_size: size of the encoder input
        _decoder_input_size: size of the encoder_output

    Methods:
        load_images_and_process: generates trajectories by sampling from pedestrian sequences
        get_data_slices: generate tracks for training/testing
        create_lstm_model: a helper function for creating conv LSTM unit
        pie_convlstm_encdec: generates intention prediction model
        train: trains the model
        test_chunk: tests the model (chunks the test cases for memory efficiency)
    """
    def __init__(self,
                 num_hidden_units=128,
                 regularizer_val=0.001,
                 activation='tanh',
                 lstm_dropout=0.4,
                 lstm_recurrent_dropout=0.2,
                 convlstm_num_filters=64,
                 convlstm_kernel_size=2):

        # Network parameters
        self._num_hidden_units = num_hidden_units
        #self._bias_initializer = 'zeros' # 'zeros' or 'ones'
        #self._output_activation = 'sigmoid'
        self.reg_value = regularizer_val
        self._kernel_regularizer = regularizers.l2(regularizer_val)
        self._recurrent_regularizer = regularizers.l2(regularizer_val)
        self._bias_regularizer = regularizers.l2(regularizer_val)
        self._activation = activation

        # Encoder
        self._lstm_dropout = lstm_dropout
        self._lstm_recurrent_dropout = lstm_recurrent_dropout

        # conv unit parameters
        self._convlstm_num_filters = convlstm_num_filters
        self._convlstm_kernel_size = convlstm_kernel_size

        #self._encoder_dense_output_size = 1 # set this only for single lstm unit
        self._encoder_input_size = 4  # decided on run time according to data

        self._decoder_dense_output_size = 1
        self._decoder_input_size = 4  # decided on run time according to data

        # Data properties
        #self._batch_size = 128  # this will be set at train time

        self._model_name = 'convlstm_encdec'

    def get_path(self,
                 type_save='models', # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=os.environ['PIE_PATH'] + '/data/'):
        """
        A path generator method for saving model and config data. Creates directories
        as needed.
        :param type_save: Specifies whether data or model is saved.
        :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
        :param model_name: model name (either trained convlstm_encdec model or vgg16)
        :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
        :param data_subset: train, test or val
        :param data_type: type of the data (e.g. features_context_pad_resize)
        :param save_root_folder: The root folder for saved data.
        :return: The full path for the save folder
        """
        assert(type_save in ['models', 'data'])
        if data_type != '':
            assert(any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def get_model_config(self):
        """
        Returns a dictionary containing model configuration.
        """

        config = dict()
        # Network parameters
        config['num_hidden'] = self._num_hidden_units
        #config['bias_init'] = self._bias_initializer
        config['reg_value'] = self.reg_value
        config['activation'] = self._activation
        config['sequence_length'] = self._sequence_length
        config['lstm_dropout'] = self._lstm_dropout
        config['lstm_recurrent_dropout'] = self._lstm_recurrent_dropout
        config['convlstm_num_filters'] = self._convlstm_num_filters
        config['convlstm_kernel_size'] = self._convlstm_kernel_size

        config['encoder_input_size'] = self._encoder_input_size

        config['decoder_input_size'] = self._decoder_input_size
        config['decoder_dense_output_size'] = self._decoder_dense_output_size

        # Set the input sizes
        config['encoder_seq_length'] = self._encoder_seq_length
        config['decoder_seq_length'] = self._decoder_seq_length

        print(config)
        return config

    def load_model_config(self, config):
        """
        Copy config information from the dictionary for testing
        """
        # Network parameters
        self._num_hidden_units = config['num_hidden']

        self.reg_value = config['reg_value']
        self._activation = config['activation']
        self._encoder_input_size = config['encoder_input_size']
        self._encoder_seq_length = config['encoder_seq_length']
        self._sequence_length = config['sequence_length']

        self._lstm_dropout = config['lstm_dropout']
        self._lstm_recurrent_dropout = config['lstm_recurrent_dropout']
        self._convlstm_num_filters = config['convlstm_num_filters']
        self._convlstm_kernel_size = config['convlstm_kernel_size']

        self._encoder_input_size = config['decoder_input_size']
        self._decoder_input_size = config['decoder_input_size']
        self._decoder_dense_output_size = config['decoder_dense_output_size']
        self._decoder_seq_length = config['decoder_seq_length']

    def load_images_and_process(self,
                                img_sequences,
                                bbox_sequences,
                                ped_ids,
                                save_path,
                                data_type='train',
                                regen_pkl=False):
        """
        Generates image features for convLSTM input. The images are first
        cropped to 1.5x the size of the bounding box, padded and resized to
        (224, 224) and fed into pretrained VGG16.
        :param img_sequences: a list of frame names
        :param bbox_sequences: a list of corresponding bounding boxes
        :ped_ids: a list of pedestrian ids associated with the sequences
        :save_path: path to save the precomputed features
        :data_type: train/val/test data set
        :regen_pkl: if set to True overwrites previously saved features
        :return: a list of image features
        """
        # load the feature files if exists
        print("Generating {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_type, save_path))
        try:
            convnet = self.context_model
        except:
            raise Exception("No context model is defined")

        sequences = []
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_sequences[i], pid):
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
                
                if os.path.exists(img_save_path) and not regen_pkl:
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                else:
                    img_data = load_img(imp)
                    bbox = jitter_bbox(imp, [b],'enlarge', 2)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    bbox = list(map(int,bbox[0:4]))
                    cropped_image = img_data.crop(bbox)
                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)                        
                    image_array = img_to_array(img_data)
                    preprocessed_img = vgg16.preprocess_input(image_array)
                    expanded_img = np.expand_dims(preprocessed_img, axis=0)
                    img_features = convnet.predict(expanded_img)
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)
                img_features = np.squeeze(img_features)
                img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        return sequences

    def get_tracks(self, dataset, data_type, seq_length, overlap):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        :param data_type: types of data for encoder/decoder input
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: a dictionary containing sampled tracks for each data modality
        """
        overlap_stride = seq_length if overlap == 0 else \
        int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d_types = []
        for k in data_type.keys():
            d_types.extend(data_type[k])
        d = {}

        if 'bbox' in d_types:
            d['bbox'] = dataset['bbox']
        if 'intention_binary' in d_types:
            d['intention_binary'] = dataset['intention_binary']
        if 'intention_prob' in d_types:
            d['intention_prob'] = dataset['intention_prob']

        bboxes = dataset['bbox'].copy()
        images = dataset['image'].copy()
        ped_ids = dataset['ped_id'].copy()

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i+seq_length] for i in\
                             range(0,len(track)\
                            - seq_length + 1, overlap_stride)])
            d[k] = tracks

        pid = []
        for p in ped_ids:
            pid.extend([p[i:i+seq_length] for i in\
                         range(0,len(p)\
                        - seq_length + 1, overlap_stride)])
        ped_ids = pid

        im = []
        for img in images:
            im.extend([img[i:i+seq_length] for i in\
                         range(0,len(img)\
                        - seq_length + 1, overlap_stride)])
        images = im

        bb = []
        for bbox in bboxes:
            bb.extend([bbox[i:i+seq_length] for i in\
                         range(0,len(bbox)\
                        - seq_length + 1, overlap_stride)])

        bboxes = bb
        return d, images, bboxes, ped_ids

    def concat_data(self, data, data_type):
        """
        Concatenates different types of data specified by data_type.
        Creats dummy data if no data type is specified
        :param data_type: type of data (e.g. bbox)
        """
        if not data_type:
            return []
        # if more than one data type is specified, they are concatenated
        d = []
        for dt in data_type:
            d.append(np.array(data[dt]))
        if len(d) > 1:
            d = np.concatenate(d, axis=2)
        else:
            d = d[0]
        return d

    def get_train_val_data(self, data, data_type, seq_length, overlap):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """
        tracks, images, bboxes, ped_ids = self.get_tracks(data, data_type, seq_length, overlap)

        # Generate observation data input to encoder
        encoder_input = self.concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = self.concat_data(tracks, data_type['decoder_input_type'])
        output = self.concat_data(tracks, data_type['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)
        # Create context model
        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'output': output}

    def get_test_data(self, data, train_params, seq_length, overlap):
        """
        A helper function for test data generation that preprocesses the images, combines
        different representations required as inputs to encoder and decoder, as well as 
        ground truth and returns them as a unified list representation.
        :param data: A dictionary of data types
        :param train_params: Training parameters defining the type of 
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """
        tracks, images, bboxes, ped_ids = self.get_tracks(data,
                                                            train_params['data_type'],
                                                            seq_length,
                                                            overlap)

        # Generate observation data input to encoder
        encoder_input = self.concat_data(tracks, train_params['data_type']['encoder_input_type'])
        decoder_input = self.concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = self.concat_data(tracks, train_params['data_type']['output_type'])
        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)

        # Create context model
        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')                                                

        test_img = self.load_images_and_process(images,
                                                bboxes,
                                                ped_ids,
                                                data_type='test',
                                                save_path=self.get_path(type_save='data',
                                                                        data_type='features_context_pad_resize',  # images
                                                                        model_name='vgg16_none',
                                                                        data_subset='test'))
        output = output[:, 0]
        return ([test_img, decoder_input], output)

    def get_model(self, model):
        train_model = self.pie_convlstm_encdec()
        return train_model

    def create_lstm_model(self,
                          name='convlstm_encdec',
                          r_state=True,
                          r_sequence=False):
        return LSTM(units=self._num_hidden_units,
                    dropout=self._lstm_dropout,
                    recurrent_dropout=self._lstm_recurrent_dropout,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    bias_initializer='zeros',
                    kernel_regularizer=self._kernel_regularizer,
                    recurrent_regularizer=self._recurrent_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    activation=self._activation,
                    name=name)

    def pie_convlstm_encdec(self):
        '''
        Create an LSTM Encoder-Decoder model for intention estimation
        '''
        #Generate input data. the shapes is (sequence_lenght,length of flattened features)
        encoder_input=input_data=Input(shape=(self._sequence_length,) + self.context_model.output_shape[1:],
                                       name = "encoder_input")
        interm_input = encoder_input

        # Generate Encoder LSTM Unit
        encoder_model = ConvLSTM2D(filters=self._convlstm_num_filters,
                                   kernel_size=self._convlstm_kernel_size,
                                   kernel_regularizer=self._kernel_regularizer,
                                   recurrent_regularizer=self._recurrent_regularizer,
                                   bias_regularizer=self._bias_regularizer,
                                   dropout=self._lstm_dropout,
                                   recurrent_dropout=self._lstm_recurrent_dropout,
                                   return_sequences=False)(interm_input)
        encoder_output = Flatten(name='encoder_flatten')(encoder_model)

        # Generate Decoder LSTM unit
        decoder_input = Input(shape=(self._decoder_seq_length,
                                     self._decoder_input_size),
                              name='decoder_input')
        encoder_vec = RepeatVector(self._decoder_seq_length)(encoder_output)
        decoder_concat_inputs = Concatenate(axis=2)([encoder_vec, decoder_input])

        decoder_model = self.create_lstm_model(name='decoder_network',
                                               r_state = False,
                                               r_sequence=False)(decoder_concat_inputs)

        decoder_dense_output = Dense(self._decoder_dense_output_size,
                                     activation='sigmoid',
                                     name='decoder_dense')(decoder_model)

        decoder_output = decoder_dense_output

        self.train_model = Model(inputs=[encoder_input, decoder_input],
                                 outputs=decoder_output)

        self.train_model.summary()
        return self.train_model

    def train(self,
              data_train,
              data_val,
              batch_size=128,
              epochs=400,
              optimizer_type='rmsprop',
              optimizer_params={'lr': 0.00001, 'clipvalue': 0.0, 'decay': 0.0},
              loss=['binary_crossentropy'],
              metrics=['acc'],
              data_opts=''):
        """
        Training method for the model
        :param data_train: training data
        :param data_val: validation data
        :param batch_size: batch size for training
        :param epochs: number of epochs for training
        :param optimizer_params: learning rate and clipvalue for gradient clipping
        :param loss: type of loss function
        :param metrics: metrics to monitor
        :param data_opts: data generation parameters
        """

        data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                     'decoder_input_type': data_opts['decoder_input_type'],
                     'output_type': data_opts['output_type']}

        train_config = {'batch_size': batch_size,
                        'epoch': epochs,
                        'optimizer_type': optimizer_type,
                        'optimizer_params': optimizer_params,
                        'loss': loss,
                        'metrics': metrics,
                        'learning_scheduler_mode': 'plateau',
                        'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                      'step_drop_rate': 0.5,
                                                      'epochs_drop_rate': 20.0,
                                                      'plateau_patience': 5,
                                                      'min_lr': 0.0000001,
                                                      'monitor_value': 'val_loss'},
                        'model': 'convlstm_encdec',
                        'data_type': data_type,
                        'overlap': data_opts['seq_overlap_rate'],
                        'dataset': 'pie'}
        self._model_type = 'convlstm_encdec'
        seq_length = data_opts['max_size_observe']
        train_d = self.get_train_val_data(data_train, data_type, seq_length, data_opts['seq_overlap_rate'])
        val_d = self.get_train_val_data(data_val, data_type, seq_length, data_opts['seq_overlap_rate'])

        self._encoder_seq_length = train_d['decoder_input'].shape[1]
        self._decoder_seq_length = train_d['decoder_input'].shape[1]

        self._sequence_length = self._encoder_seq_length

        # Create context model
        self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')

        train_img = self.load_images_and_process(train_d['images'],
                                                 train_d['bboxes'],
                                                 train_d['ped_ids'],
                                                 data_type='train',
                                                 save_path=self.get_path(type_save='data',
                                                                         data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'], # images    
                                                                         model_name='vgg16_'+'none',
                                                                         data_subset = 'train'))
        val_img = self.load_images_and_process(val_d['images'],
                                               val_d['bboxes'],
                                               train_d['ped_ids'],
                                               data_type='val',
                                               save_path=self.get_path(type_save='data',
                                                                       data_type='features'+'_'+data_opts['crop_type']+'_'+data_opts['crop_mode'],
                                                                       model_name='vgg16_'+'none',
                                                                       data_subset='val'))

        train_model = self.pie_convlstm_encdec()

        train_d['output'] = train_d['output'][:, 0]
        val_d['output'] = val_d['output'][:, 0]

        train_data = ([train_img, train_d['decoder_input']], train_d['output'])
        val_data = ([val_img, val_d['decoder_input']], val_d['output'])

        optimizer = RMSprop(lr=optimizer_params['lr'],
                            decay=optimizer_params['decay'],
                            clipvalue=optimizer_params['clipvalue'])

        train_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print('TRAINING: loss={} metrics={}'.format(loss, metrics))

        #automatically generate model name as a time string
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        model_path, _ = self.get_path(type_save='models',
                                      model_name='convlstm_encdec',
                                      models_save_folder=model_folder_name,
                                      file_name='model.h5',
                                      save_root_folder='data')
        config_path, _ = self.get_path(type_save='models',
                                       model_name='convlstm_encdec',
                                       models_save_folder=model_folder_name,
                                       file_name='configs',
                                       save_root_folder='data')

        #Save config and training param files
        with open(config_path+'.pkl', 'wb') as fid:
            pickle.dump([self.get_model_config(),
                        train_config, data_opts],
                        fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        #Save config and training param files
        with open(config_path+'.txt', 'wt') as fid:
            fid.write("####### Data options #######\n")
            fid.write(str(data_opts))
            fid.write("\n####### Model config #######\n")
            fid.write(str(self.get_model_config()))
            fid.write("\n####### Training config #######\n")
            fid.write(str(train_config))


        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=5,
                                   verbose=1)
        checkpoint = ModelCheckpoint(filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor=train_config['learning_scheduler_params']['monitor_value'])  #, mode = 'min'
        plateau_sch = ReduceLROnPlateau(monitor=train_config['learning_scheduler_params']['monitor_value'],
                factor=train_config['learning_scheduler_params']['step_drop_rate'],
                patience=train_config['learning_scheduler_params']['plateau_patience'],
                min_lr=train_config['learning_scheduler_params']['min_lr'],
                verbose = 1)

        call_backs = [checkpoint, early_stop, plateau_sch]

        history = train_model.fit(x=train_data[0],
                                  y=train_data[1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=val_data,
                                  callbacks=call_backs,
                                  verbose=1)

        history_path, saved_files_path = self.get_path(type_save='models',
                                                       model_name='convlstm_encdec',
                                                       models_save_folder=model_folder_name,
                                                       file_name='history.pkl',
                                                       save_root_folder='data')

        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        del train_data, val_data  # clear memory
        del train_d, val_d
        
        return saved_files_path

    #split test data into chunks
    def test_chunk(self,
                   data_test,
                   data_opts='',
                   model_path='',
                   visualize=False):
            with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
                try:
                    configs = pickle.load(fid)
                except:
                    configs = pickle.load(fid, encoding='bytes')
            train_params = configs[1]
            self.load_model_config(configs[0])
                # Create context model
            self.context_model = vgg16.VGG16(input_shape=(224, 224, 3),
                                             include_top=False,
                                             weights='imagenet')
            try:
                test_model = load_model(os.path.join(model_path, 'model.h5'))

            except:
                test_model = self.get_model(train_params['model'])
                test_model.load_weights(os.path.join(model_path, 'model.h5'))
            
            test_model.summary()

            overlap = 1  # train_params ['overlap']

            test_target_data = []
            test_results = []
            ped_ids = []
            images = []
            bboxes = []

            num_samples = len(data_test['image'])

            vis_results = []

            for i in range(0, len(data_test['image']), 100):

                data_test_chunk = {}
                data_test_chunk['intention_binary'] = data_test['intention_binary'][i:min(i+100, num_samples)]
                data_test_chunk['image'] = data_test['image'][i:min(i+100,num_samples)]
                data_test_chunk['ped_id'] = data_test['ped_id'][i:min(i+100,num_samples)]
                data_test_chunk['intention_prob'] = data_test['intention_prob'][i:min(i+100,num_samples)]
                data_test_chunk['bbox'] = data_test['bbox'][i:min(i+100,num_samples)]

                test_data_chunk, test_target_data_chunk = self.get_test_data(data_test_chunk,
                                                                                    train_params,
                                                                                    self._sequence_length,
                                                                                    overlap)

                tracks, images_chunk, bboxes_chunk, ped_ids_chunk = self.get_tracks(data_test_chunk,
                                                                                           train_params['data_type'],
                                                                                           self._sequence_length,
                                                                                           overlap)


                test_results_chunk = test_model.predict(test_data_chunk,
                                                        batch_size=train_params['batch_size'],
                                                        verbose=1)

                test_target_data.extend(test_target_data_chunk)
                test_results.extend(test_results_chunk)
                images.extend(images_chunk)
                ped_ids.extend(ped_ids_chunk)
                bboxes.extend(bboxes_chunk)
                
                i = -1
                for imp, box, ped in zip(images_chunk, bboxes_chunk, ped_ids_chunk):
                    i+=1
                    vis_results.append({'imp': imp[-1], 
                                        'bbox': box[-1],
                                        'ped_id': ped[-1][0],
                                        'res': test_results_chunk[i][0],
                                        'target': test_target_data_chunk[i]})


            acc = accuracy_score(test_target_data, np.round(test_results))
            f1 = f1_score(test_target_data, np.round(test_results))

            save_results_path = os.path.join(model_path, 'ped_intents.pkl')
            if not os.path.exists(save_results_path):
                results = {'ped_id': ped_ids,
                           'images': images,
                           'results': test_results,
                           'gt': test_target_data}
                with open(save_results_path, 'wb') as fid:
                    pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)
            return acc, f1
