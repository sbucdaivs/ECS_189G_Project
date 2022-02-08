import pickle
import matplotlib.pyplot as plt

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        # f = open('../../data/stage_3_data/MNIST', 'rb')
        data = pickle.load(f)
        f.close()
        return data
