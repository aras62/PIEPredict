import os

from jaad_utilities.jaad_data import JAAD
from pie_utilities.pie_data import PIE
from pie_predict import PIEPredict
from pie_intent import PIEIntent

#train models with data up to critical point
#train_test = 0 (train only), 1 (train-test), 2 (test only)


def train_intent(data_opts, train_test=1):

        t = PIEIntent(num_hidden_units=128,
                      regularizer_val=0.001,
                      lstm_dropout=0.4,
                      lstm_recurrent_dropout=0.2,
                      convlstm_num_filters=64,
                      convlstm_kernel_size=2)

        model_opts = {'crop_type': 'context',  # crop 2x bbox
                      'crop_mode': 'pad_resize',  # pad with 0s and resize
                      'max_size_observe': 15,  # length of observations
                      'max_size_predict': 5,  # length of prediction                    # occluded frames
                      'enc_input_type': [],
                      'dec_input_type': ['bbox'],
                      'prediction_type': ['intention_binary']}


        saved_files_path = ''
        dataset = 'pie'

        test_name = 'PIE_intent_context_loc'
        model_name = 'convlstm_encdec'

        imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])        
        #for test
        root = os.path.join(os.environ['PIE_PATH'], 'data', 'models', model_name, dataset, test_name)


        if train_test < 2:    # Train
                beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
                beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

                beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
                beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

                saved_files_path = t.train(data_train=beh_seq_train,
                                           data_val=beh_seq_val,
                                           epochs=400,
                                           loss=['binary_crossentropy'],
                                           metrics=['accuracy'],
                                           batch_size=128,
                                           optimizer_type='rmsprop',
                                           data_opts=data_opts)

                print(data_opts['seq_overlap_rate'])

        if train_test > 0:    # Test
                if saved_files_path == '':
                        saved_files_path = root
                beh_seq_test = imdb.generate_data_trajectory_sequence('test', **model_opts)
                acc, f1 = t.test_chunk(beh_seq_test, model_opts, saved_files_path, False)

                K.clear_session()
                tf.reset_default_graph()
                return acc, f1, saved_files_path


def train_predict(data_opts, train_test=1):

    pie_predict_speed = PIEPredict()
    pie_predict_traj = PIEPredict()

    pie_path = os.environ.copy()['PIE_PATH']
    jaad_path = os.environ.copy()['JAAD_PATH']

    dataset = 'pie'
    if dataset == 'pie':
            imdb = PIE(data_path=pie_path)
    elif dataset == 'jaad':
            imdb = JAAD(data_path=jaad_path)

    model_opts = {'normalize_bbox': True,
                  'track_overlap': 0.5,    # 0.8 for jaad, 0.5 pie
                  'observe_length': 15,
                  'predict_length': 45,
                  'enc_input_type': ['bbox'],
                  'dec_input_type': [],
                  'prediction_type': ['bbox']}


    #train_test = 2 # 0 train, 1 train-test, 2 test
    saved_files_path = ''    # model test name

    if train_test < 2:
            beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
            beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
            saved_files_path = t.train(beh_seq_train, beh_seq_val, epochs=2, **model_opts)
    if train_test > 0:
            beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
            t.test(beh_seq_test, saved_files_path)
            # t.test_final(beh_seq_test, saved_files_path)


def main(train_test=2):

    intent_data_opts = {'fstride': 1,
                        'sample_type': 'all',  # 'beh'. Only for JAAD dataset
                        'height_rng': [0, float('inf')],
                        'squarify_ratio': 0,
                        'data_split_type': 'default',  #  kfold, random, default
                        'seq_type': 'intention', #  crossing , intention
                        'min_track_size': 0, #  discard tracks that are shorter
                        'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                        }

    train_intent(intent_data_opts, train_test=train_test)

    predict_data_opts = {'fstride': 1,
                         'sample_type': 'all',    # 'beh'. Only for JAAD dataset
                         'subset': 'default',
                         'height_rng': [0, float('inf')],
                         'squarify_ratio': 0,
                         'data_split_type': 'default',    # kfold, random, default
                         'seq_type': 'trajectory',
                         'min_track_size': 61,
                         'random_params': {'ratios': None,
                                           'val_data': True,
                                           'regen_data': True},
                         'kfold_params': {'num_folds': 5, 'fold': 1}}

    train_predict(predict_data_opts, train_test=train_test)

if __name__ == '__main__':
    if len(sys.argv) > 0:
        try:
            arg = int(sys.argv[1])
            main(int(sys.argv[1]))
        except ValueError:
            raise ValueError('Usage: python train_test_intent.py TRAIN_TEST_OPT\n'
                             '\t\t\t\tTRAIN_TEST_OPT=0 - train only\n'
                             '\t\t\t\tTRAIN_TEST_OPT=1 - train and test\n'
                             '\t\t\t\tTRAIN_TEST_OPT=2 - test only\n')
    else:
        main()