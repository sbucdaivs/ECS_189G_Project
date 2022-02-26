from code_src.stage_2_code.Dataset_Loader import Dataset_Loader
from code_src.stage_2_code.Method_MLP import Method_MLP
from code_src.stage_2_code.Result_Saver import Result_Saver
from code_src.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code_src.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)   # TODO: Why does it need this?
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    # training set
    train_data_obj = Dataset_Loader('MNIST_training', 'Handwritten digits.')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    # train_data_obj.dataset_source_file_name = 'train.csv'
    train_data_obj.dataset_source_file_name = 'train_small.csv'

    # testing set
    test_data_obj = Dataset_Loader('MNIST_testing', 'Handwritten digits.')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'


    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score, precision, std_pre, recall, std_recall, f1, std_f1 = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('Precision:'+str(precision) +  ' +/- ' + str(std_pre))
    print('Recall:' + str(recall) + ' +/- ' + str(std_recall))
    print('F1 score:' + str(f1) + ' +/- ' + str(std_f1))
    print('************ Finish ************')
    # ------------------------------------------------------
    # TODO: Do overall testing on test.csv

    scores = setting_obj.eval_test(test_data_obj)
    accuracy = scores[0]
    precision = scores[1][0]
    recall = scores[1][1]
    f1 = scores[1][2]
    print("MLP Accuracy on test dataset: {}".format(accuracy))
    print("Precision:", precision, "Recall:", recall, "F1 score:", f1)
