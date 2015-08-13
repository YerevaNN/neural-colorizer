import cPickle as pickle
import numpy as np

def load_cifar_train_data():
    """
        train_images format is: (index, channel, height, width)
        train_labels format is: (index, label)
    """
    train_images = []
    train_labels = np.array([])
    
    for iter in range(1, 6):
        batch_file_name = 'data/cifar-10-batches-py/data_batch_' + str(iter)
        batch_file = open(batch_file_name, 'r')
        dict = pickle.load(batch_file)
        batch_file.close()
        
        batch_images = dict['data']
        batch_labels = dict['labels']
        
        train_images.append(batch_images)
        train_labels = np.append(train_labels, batch_labels)
    
    train_images = np.vstack(train_images) / 256.0
    train_images = train_images.reshape(train_images.shape[0], 3, 32, 32)

    return (train_images, train_labels)

def load_cifar_val_data():
    """
        val_images format is: (index, channel, height, width)
        val_labels format is: (index, label)
    """
    file_name = 'data/cifar-10-batches-py/test_batch'
    file = open(file_name, 'r')
    dict = pickle.load(file)
    file.close()
        
    val_images = dict['data'] / 256.0
    val_images = val_images.reshape(val_images.shape[0], 3, 32, 32)
    val_labels = np.array(dict['labels'])

    return (val_images, val_labels)