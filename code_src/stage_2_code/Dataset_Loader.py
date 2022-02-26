'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_src.base_class.dataset import dataset
import csv


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        f_read = csv.reader(f)
        for line in f_read:
            X.append([int(i) for i in line[1:]])
            y.append(int(line[0]))
        f.close()
        return {'X': X, 'y': y}
