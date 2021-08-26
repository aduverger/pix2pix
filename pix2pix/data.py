import os 
import glob
import imageio
import numpy as np
from tensorflow import convert_to_tensor
from tensorflow.data import Dataset

def load_data(host='drive', dataset='facades'):
    """Return three tensors of datas (train, val, test), given a certain dataset name (e.g. 'facades).

    Args:
        host (str): Where the dataset is host, i.e. 'drive' or 'local'. Defaults to 'drive'.
        dataset (str): Dataset to load. Defaults to 'facades'.

    Returns:
        data_train, data_val, data_test (tf.Tensor, dtype='float32'): The three datasets (train, val, test) as tensors with 'float32' type
    """
    
    if host == 'drive':
        directory = '/content/drive/MyDrive/pix2pix/datasets'
        if dataset == 'facades':
            directory += '/resized'
    else: #if host == 'local'
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        if dataset == 'facades':
            directory += '/facades'
    #TODO create directories for other datasets and check validity of host and dataset
    
    data_train = []
    data_val = []
    data_test = []
    for im_path in glob.glob(directory+"/train/*.jpg"):
        data_train.append(imageio.core.asarray(imageio.imread(im_path)))

    for im_path in glob.glob(directory+"/val/*.jpg"):
        data_val.append(imageio.core.asarray(imageio.imread(im_path)))

    for im_path in glob.glob(directory+"/test/*.jpg"):
        data_test.append(imageio.core.asarray(imageio.imread(im_path)))

    data_train = convert_to_tensor(np.array(data_train), dtype='float32')
    data_val = convert_to_tensor(np.array(data_val), dtype='float32')
    data_test = convert_to_tensor(np.array(data_test), dtype='float32')
    
    return data_train, data_val, data_test

def normalize_data(data):
    """Normalize data between -1 and 1, considering that the input data values are between 0 and 256 (RGB).

    Args:
        data (tf.Tensor): Data to normalize.

    Returns:
        data (tf.Tensor): Normalized data.
    """
    
    data = (data - 127.5) / 127.5
    return data
    
def split_images(data):
    """Split images from data (with shapes (256, 512, 3)) into two images (with shapes (256, 256, 3)).

    Args:
        data (tf.Tensor): Data to split.

    Returns:
        X, Y (tf.tensor): Data split between left images (X) and right images (Y).
    """
    
    X = data[:, :, 256:]
    Y = data[:, :, :256]
    return X, Y
    
def create_dataset(X, batch_size=16) :
    """Create a tf.Dataset from a tf.tensor, given a batch size.

    Args:
        X (tf.Tensor): Data from which we want to create a dataset.
        batch_size (int): The size for the batch to create. Defaults to 16.

    Returns:
        X_ds (tf.Dataset)
    """
    
    return Dataset.from_tensor_slices(X).batch(batch_size)

def get_facades_datasets(host='drive', batch_size=16):
    """Complete function to get the datasets you need for training a pix2pix model on the facades dataset

    Args:
        host (str): Where the dataset is host, i.e. 'drive' or 'local'. Defaults to 'drive'.
        batch_size (int): The size for the batch of the datasets. Defaults to 16.

    Returns:
        X_ds_train, ..., Y_ds_test : The 6 tf.Dataset needed - (X, Y) from (train, val, test)
    """
    
    data_train, data_val, data_test = load_data(host, dataset='facades')
    data_train, data_val, data_test = normalize_data(data_train), normalize_data(data_val), normalize_data(data_test)
    X_train, Y_train = split_images(data_train)
    X_val, Y_val = split_images(data_val)
    X_test, Y_test = split_images(data_test)
    
    X_ds_train, Y_ds_train = create_dataset(X_train, batch_size), create_dataset(Y_train, batch_size)
    X_ds_val, Y_ds_val = create_dataset(X_val, batch_size), create_dataset(Y_val, batch_size)
    X_ds_test, Y_ds_test = create_dataset(X_test, batch_size), create_dataset(Y_test, batch_size)
    
    return X_ds_train, X_ds_val, X_ds_test, Y_ds_train, Y_ds_val, Y_ds_test
    
if __name__ == "__main__":
    X_ds_train, X_ds_val, X_ds_test, Y_ds_train, Y_ds_val, Y_ds_test = get_facades_datasets(host='local')
    print(type(X_ds_test))
    print(Y_ds_val)
