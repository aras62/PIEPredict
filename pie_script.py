import os

from jaad_utilites.jaad_data import JAAD
from pie_utilities.pie_data import PIE
from iccv.pie_predict import PIEPredict

data_opts = {'fstride': 1,
             'sample_type': 'all',  # 'beh'. Only for JAAD dataset
             'subset': 'default',
             'height_rng': [0, float('inf')],
             'squarify_ratio': 0,
             'data_split_type': 'default',  # kfold, random, default
             'seq_type': 'trajectory',
             'min_track_size': 61,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}

t = PIEPredict()
pie_path = os.environ.copy()['PIE_PATH']
jaad_path = os.environ.copy()['JAAD_PATH']

dataset = 'pie'
if dataset == 'pie':
    imdb = PIE(data_path=pie_path)
elif dataset == 'jaad':
    imdb = JAAD(data_path=jaad_path)

model_opts = {'normalize_bbox': True,
              'track_overlap': 0.5,  # 0.8 for jaad, 0.5 pie
              'observe_length': 15,
              'predict_length': 45,
              'enc_input_type': ['bbox'],
              'dec_input_type': [],
              'prediction_type': ['bbox']}

train_test = 2 # 0 train, 1 train-test, 2 test
saved_files_path = ''  # model test name

if train_test < 2:
    beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    saved_files_path = t.train(beh_seq_train, beh_seq_val, epochs=2, **model_opts)
if train_test > 0:
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    t.test(beh_seq_test, saved_files_path)
    # t.test_final(beh_seq_test, saved_files_path)
