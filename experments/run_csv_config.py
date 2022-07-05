import sys

sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
# from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM, DeepFM_fft, AutoInt, AutoInt_fft, DCN, DCN_fft
from fuxictr.pytorch.torch_utils import seed_everything

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--model', type=str, default='AutoInt_fft')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Load params from config files
    config_dir = 'demo_config'
    # experiment_id = 'AutoInt_fft'  # correponds to csv input `taobao_tiny`
    experiment_id = args.model
    params = load_config(config_dir, experiment_id)
    params['gpu'] = args.gpu
    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set feature_encoder that defines how to preprocess data
    if params['dataset_id'] in ['taobao_tiny', 'taobao_tiny_h5']:
        FeatureEncoder = datasets.taobao.FeatureEncoder
    elif params['dataset_id'] in ['avazu_x4', 'avazu_x4_h5']:
        FeatureEncoder = datasets.avazu.FeatureEncoder
    elif params['dataset_id'] in ['criteo_x4', 'criteo_x4_h5']:
        FeatureEncoder = datasets.criteo.FeatureEncoder
    feature_encoder = FeatureEncoder(params['feature_cols'],
                                     params['label_col'],
                                     dataset_id=params['dataset_id'],
                                     data_root=params["data_root"])

    # Build dataset from csv to h5
    datasets.build_dataset(feature_encoder,
                           train_data=params["train_data"],
                           valid_data=params["valid_data"],
                           test_data=params["test_data"])

    # Get feature_map that defines feature specs
    feature_map = feature_encoder.feature_map

    # Get train and validation data generator from h5
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    train_gen, valid_gen = datasets.h5_generator(feature_map,
                                                 stage='train',
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])

    # Model initialization and fitting
    if experiment_id == 'DeepFM_base':
        model = DeepFM(feature_encoder.feature_map, **params)
    elif experiment_id == 'DeepFM_fft':
        model = DeepFM_fft(feature_encoder.feature_map, **params)
    if experiment_id == 'AutoInt_base':
        model = AutoInt(feature_encoder.feature_map, **params)
    elif experiment_id == 'AutoInt_fft':
        model = AutoInt_fft(feature_encoder.feature_map, **params)
    if experiment_id == 'DCN_base':
        model = DCN(feature_encoder.feature_map, **params)
    elif experiment_id == 'DCN_fft':
        model = DCN_fft(feature_encoder.feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)  # reload the best checkpoint

    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    logging.info('***** validation results *****')
    test_gen = datasets.h5_generator(feature_map,
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)
