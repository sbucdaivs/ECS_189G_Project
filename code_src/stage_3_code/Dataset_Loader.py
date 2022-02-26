import pickle
import matplotlib.pyplot as plt

from code_src.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data: dict = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    data_type = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def pre_load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        self.data = pickle.load(f)
        f.close()

    # TODO: update below to reflect different data structure
    def load(self, dType = 'train'):
        print('loading data...')
        X = []
        y = []
        line: dict
        for line in self.data[dType]:
            X.append(line['image'])
            y.append(line['label'])


        return {'X': X, 'y': y}
