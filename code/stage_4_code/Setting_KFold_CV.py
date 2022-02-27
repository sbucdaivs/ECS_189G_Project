'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from code.stage_4_code.Method_RNN import Method_RNN
from code.base_class.setting import setting
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

class Setting_KFold_CV(setting):
    fold = 3

    def load_run_save_evaluate_plot(self):
        # load dataset
        loaded_data = self.dataset.load()['train']
        # print(loaded_data['X'][0])
        kf = KFold(n_splits=self.fold, shuffle=True)

        fold_count = 0
        score_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        epoch_list = []
        loss_list = []
        for train_index, test_index in kf.split(loaded_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = np.array(loaded_data['X'])[train_index], np.array(loaded_data['X'])[test_index]
            y_train, y_test = np.array(loaded_data['y'])[train_index], np.array(loaded_data['y'])[test_index]

            # create Tensor datasets
            train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

            # dataloaders
            batch_size = 2
            # make sure to SHUFFLE your data
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
            test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


            # run MethodModule
            self.method.data = {'train': train_loader, 'test': test_loader}
            # self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()

    #         # save raw ResultModule
    #         self.result.data = learned_result
    #         self.result.fold_count = fold_count
    #         self.result.save()
    #
    #         self.evaluate.data = learned_result
    #         scores = self.evaluate.evaluate()
    #         score_list.append(scores[0])
    #         precision_list.append(scores[1])
    #         recall_list.append(scores[2])
    #         f1_list.append(scores[3])
    #
    #     # plot_training_convergence()
    #
    #     return np.mean(score_list), np.std(score_list), np.mean(precision_list), np.std(precision_list), \
    #            np.mean(recall_list), np.std(recall_list), np.mean(f1_list), np.std(f1_list)
    #
    # def eval_test(self, dataset):
    #     test_data = dataset.load('test')
    #     y_pred = self.method.test(test_data['X'])
    #     return accuracy_score(test_data['y'], y_pred), \
    #            precision_recall_fscore_support(test_data['y'], y_pred, average='weighted')


# def plot_training_convergence():
#     epoch = Method_CNN.max_epoch
#     plt.plot(Method_CNN.epoch_list[0:epoch], Method_CNN.loss_list[0:epoch], "r", label="Fold 1")
#     plt.plot(Method_CNN.epoch_list[epoch:2 * epoch + 1], Method_CNN.loss_list[epoch:2 * epoch + 1], "g",
#              label="Fold 2")
#     plt.plot(Method_CNN.epoch_list[2 * epoch + 1:], Method_CNN.loss_list[2 * epoch + 1:], "b", label="Fold 3")
#     plt.legend(loc="upper right")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.title("Training Convergence Plot")
#     plt.savefig("Training Convergence Plot.png")
#     plt.show()
