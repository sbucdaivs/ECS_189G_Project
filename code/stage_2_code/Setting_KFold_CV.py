'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        precision_list = []
        recall_list = []
        f1_list=[]
        for train_index, test_index in kf.split(loaded_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = np.array(loaded_data['X'])[train_index], np.array(loaded_data['X'])[test_index]
            y_train, y_test = np.array(loaded_data['y'])[train_index], np.array(loaded_data['y'])[test_index]
        
            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            scores = self.evaluate.evaluate()
            score_list.append(scores[0])
            precision_list.append(scores[1])
            recall_list.append(scores[2])
            f1_list.append(scores[3])

        return np.mean(score_list), np.std(score_list), np.mean(precision_list), np.std(precision_list),\
            np.mean(recall_list), np.std(recall_list), np.mean(f1_list), np.std(f1_list)

    def eval_test(self, test_dataset):
        test_data = test_dataset.load()
        y_pred = self.method.test(test_data['X'])
        return accuracy_score(test_data['y'], y_pred), \
               precision_recall_fscore_support(test_data['y'], y_pred, average='weighted')
