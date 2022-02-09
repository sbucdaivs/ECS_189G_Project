import pickle
import matplotlib.pyplot as plt

if 1:
    # f = open('../../data/stage_3_data/MNIST', 'rb')  # or change MNIST to other dataset names
    f = open('./data/stage_3_data/MNIST', 'rb')  # or change MNIST to other dataset names
    data = pickle.load(f)
    f.close()

    print('training set size:', len(data['train']), 'testing set size:', len(data['test']))

    #
    # for pair in data['train']:
    # #for pair in data['test']:
    #     plt.imshow(pair['image'], cmap="Greys")
    #     plt.show()
    #     print(pair['label'])
