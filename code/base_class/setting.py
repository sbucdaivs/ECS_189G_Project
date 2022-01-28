'''
Base SettingModule class for all experiment settings
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc

#-----------------------------------------------------
class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''
    
    setting_name = None
    setting_description = None

    fold = None
    
    dataset = None
    method = None
    result = None
    evaluate = None


    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription
    
    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)
        print('k: ', self.fold, ',epoch: ', self.method.max_epoch, ', learning rate: ', self.method.learning_rate) # TODO: Formatting


    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
