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


def vis(x):
    pass


if __name__ == '__main__':
    config_dir = 'demo_config'
    # experiment_id = 'AutoInt_fft'  # correponds to csv input `taobao_tiny`
    experiment_id = 'AutoInt_fft'
    params = load_config(config_dir, experiment_id)
    FeatureEncoder = datasets.taobao.FeatureEncoder
    # FeatureEncoder = datasets.avazu.FeatureEncoder
    # FeatureEncoder = datasets.criteo.FeatureEncoder
    feature_encoder = FeatureEncoder(params['feature_cols'],
                                     params['label_col'],
                                     dataset_id=params['dataset_id'],
                                     data_root=params["data_root"])
    datasets.build_dataset(feature_encoder,
                           train_data=params["train_data"],
                           valid_data=params["valid_data"],
                           test_data=params["test_data"])
    model = AutoInt_fft(feature_encoder.feature_map, **params)
    model.load_weights(model.checkpoint)
    x = model.filterlayer.complex_weight.to('cpu').data.numpy()
    vis(x[0])
