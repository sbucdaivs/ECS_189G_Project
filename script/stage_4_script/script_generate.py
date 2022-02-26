from code.stage_4_code.Dataset_Loader import Dataset_Loader
# from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Method_Generate import Method_Generate
from code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
# from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- objection initialization section ---------------
    # training set
    data_obj = Dataset_Loader('Jokes', '-', type='generation')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.load()

    method_obj = Method_Generate('Convolutional Neuron Network', '', data_obj.num_words)