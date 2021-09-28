import os
import glob
import imageio
import numpy as np
from tensorflow import convert_to_tensor
from tensorflow import data, io, image, stack, cast, shape, float32
import tensorflow.random as tf_random

def load_data(host='drive', dataset='facades'):
    """Return three tensors of datas (train, val, test), given a certain dataset name (e.g. 'facades).

    Args:
        host (str): Where the dataset is host, i.e. 'drive' or 'local'.
                    Can also be the direct file path to the dataset.
                    Defaults to 'drive'.
        dataset (str): Dataset to load. Defaults to 'facades'.

    Returns:
        data_train, data_val, data_test (tf.Tensor, dtype='float32'): The three datasets (train, val, test) as tensors with 'float32' type
    """

    if host == 'drive':
        directory = '/content/drive/MyDrive/pix2pix/datasets'
        if dataset == 'facades':
            directory += '/resized'
    elif host == 'local':
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        if dataset == 'facades':
            directory += '/facades'
    else:
        directory = host
    #TODO create directories for other datasets and check validity of host and dataset

    data_train = []
    data_val = []
    data_test = []

    train_pathes = glob.glob(directory+"/train/*.jpg")
    val_pathes   = glob.glob(directory+"/val/*.jpg")
    test_pathes  = glob.glob(directory+"/test/*.jpg")


    # Process train pathes
    ite_count, ite_total = 0, len(train_pathes)  # Track progress in imports
    for im_path in train_pathes:
        if ite_count % (ite_total // 10) == 0:
            print(f'   Import train in progress... {int(ite_count * 100 / ite_total):02d}%', end='\r')
        data_train.append(imageio.core.asarray(imageio.imread(im_path)))
        ite_count += 1
    print('Import train  -------------- DONE!')


    # Process val pathes
    ite_count, ite_total = 0, len(val_pathes)
    for im_path in val_pathes:
        if ite_count % (ite_total // 10) == 0:
            print(f'   Import val in progress... {int(ite_count * 100 / ite_total):02d}%', end='\r')
        data_val.append(imageio.core.asarray(imageio.imread(im_path)))
        ite_count += 1
    print('Import val    -------------- DONE!')


    # Process test pathes
    ite_count, ite_total = 0, len(test_pathes)
    for im_path in test_pathes:
        if ite_count % (ite_total // 10) == 0:
            print(f'   Import test progress... {int(ite_count * 100 / ite_total):02d}%', end='\r')
        data_test.append(imageio.core.asarray(imageio.imread(im_path)))
        ite_count += 1
    print('Import test   -------------- DONE!')

    data_train = convert_to_tensor(np.array(data_train), dtype='float32')
    data_val = convert_to_tensor(np.array(data_val), dtype='float32')
    data_test = convert_to_tensor(np.array(data_test), dtype='float32')

    return data_train, data_val, data_test

def normalize_data(data):
    """Normalize data between -1 and 1, considering that the input data values are between 0 and 255 (RGB).

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

    return data.Dataset.from_tensor_slices(X).batch(batch_size)

def get_facades_datasets(host='drive', batch_size=16):
    """Complete function to get the datasets you need for training a pix2pix model on the facades dataset

    Args:
        host (str): Where the dataset is host, i.e. 'drive' or 'local'. Defaults to 'drive'.
        batch_size (int): The size for the batch of the datasets. Defaults to 16.

    Returns:
        paint_ds_train, ..., real_ds_test : The 6 tf.Dataset needed - (X, Y) from (train, val, test)
    """

    data_train, data_val, data_test = load_data(host, dataset='facades')
    data_train, data_val, data_test = normalize_data(data_train), normalize_data(data_val), normalize_data(data_test)
    paint_train, real_train = split_images(data_train)
    paint_val, real_val = split_images(data_val)
    paint_test, real_test = split_images(data_test)

    paint_ds_train, real_ds_train = create_dataset(paint_train, batch_size), create_dataset(real_train, batch_size)
    paint_ds_val, real_ds_val = create_dataset(paint_val, batch_size), create_dataset(real_val, batch_size)
    paint_ds_test, real_ds_test = create_dataset(paint_test, batch_size), create_dataset(real_test, batch_size)

    return paint_ds_train, paint_ds_val, paint_ds_test, real_ds_train, real_ds_val, real_ds_test


def load_and_split_image(image_path):
    '''
        Load an image from image_path and split between paint and real images
    '''
    image_ = io.read_file(image_path)
    image_ = image.decode_jpeg(image_)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = shape(image_)[1]
    w = w // 2
    paint_image = image_[:, w:, :]
    real_image = image_[:, :w, :]

    # Convert both images to float32 tensors
    paint_image = cast(paint_image, float32)
    real_image = cast(real_image, float32)

    paint_image = (paint_image - 127.5) / 127.5
    real_image = (real_image - 127.5) / 127.5

    return paint_image, real_image


def get_dataset(host='drive', dataset='facades', batch_size=1):
    """Return three preprocessed tensors of datas (train, val, test), given a certain dataset name (e.g. 'facades').

        Args:
            host (str): Where the dataset is host, i.e. 'drive' or 'local'.
                        Can also be the direct file path to the dataset.
                        Defaults to 'drive'.
            dataset (str): Dataset to load. Defaults to 'facades'.
            batch_size (int): Size of the batches. Defaults to 1.

        Returns:
            data_train, data_val, data_test (tf.Tensor, dtype='float32'): The three datasets (train, val, test) as tensors with 'float32' type
    """

    if host == 'drive':
        directory = '/content/drive/MyDrive/pix2pix/datasets'
        if dataset == 'facades':
            directory += '/resized'
    elif host == 'local':
        directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'datasets')
        if dataset == 'facades':
            directory += '/facades'
    else:
        directory = host

    train_dataset = data.Dataset.list_files(directory + "/train/*.jpg")
    train_dataset = train_dataset.map(load_and_split_image,
                                      num_parallel_calls=data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    print('Importing train DONE!')

    val_dataset = data.Dataset.list_files(directory + "/val/*.jpg")
    val_dataset = val_dataset.map(load_and_split_image,
                                  num_parallel_calls=data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    print('Importing val DONE!')

    test_dataset = data.Dataset.list_files(directory + "/test/*.jpg")
    test_dataset = test_dataset.map(load_and_split_image,
                                    num_parallel_calls=data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    print('Importing test DONE!')

    return train_dataset, val_dataset, test_dataset


def resize(input_image, real_image, height, width):
    input_image = image.resize(input_image, [height, width],
                               method=image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = image.resize(real_image, [height, width],
                              method=image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    batch_size = input_image.shape[0]
    stacked_image = stack([input_image, real_image], axis=0)
    #print(stacked_image.shape)
    cropped_image = image.random_crop(stacked_image,
                                      size=[2, batch_size, 256, 256, 3])

    return cropped_image[0], cropped_image[1]


def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf_random.uniform(()) > 0.5:
        # Random mirroring
        input_image = image.flip_left_right(input_image)
        real_image = image.flip_left_right(real_image)

    return input_image, real_image


if __name__ == "__main__":
    train, val, test = get_dataset(host='local', dataset='facades', batch_size=8)
    paint, real = next(iter(train))
    print(paint.shape, real.shape)
