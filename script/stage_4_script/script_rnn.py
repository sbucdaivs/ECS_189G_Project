from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Recurrent Neuron Network script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    # training set
    data_obj = Dataset_Loader('IMDB', 'Movie Review')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/0_toy_dataset'
    data_obj.load()
    # print(data_obj.vocab_size)
    # testing set
    # test_data_obj = Dataset_Loader('IMDB', 'Movie Review Test')
    # test_data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/1_medium_dataset/test'
    # test_data_obj.load()

    method_obj = Method_RNN('Recurrent Neuron Network', '', data_obj.vocab_size)
    #
    # result_obj = Result_Saver('saver', '')
    # result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    # result_obj.result_destination_file_name = 'prediction_result'
    #
    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #
    # evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # # ------------------------------------------------------
    #
    # # ---- running section ---------------------------------
    # print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj)#, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate_plot()
    # print('************ Overall Performance ************')
    # print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('Precision:' + str(precision) + ' +/- ' + str(std_pre))
    # print('Recall:' + str(recall) + ' +/- ' + str(std_recall))
    # print('F1 score:' + str(f1) + ' +/- ' + str(std_f1))
    # print('************ Finish ************')
    # # ------------------------------------------------------
    #
    # scores = setting_obj.eval_test(data_obj)
    # setting_obj.classification_report(data_obj)
    # accuracy = scores[0]
    # precision = scores[1][0]
    # recall = scores[1][1]
    # f1 = scores[1][2]
    # print("CNN Accuracy on test dataset: {}".format(accuracy))
    # print("Precision:", precision, "Recall:", recall, "F1 score:", f1)
