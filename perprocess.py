# No need to run this retail pickle is already done

import cPickle
import os
import random
import hashlib
import cv2
import numpy as np

data_root = 'dataset/'
pixel_depth = 255.0
image_size = 32


def splitIntoTrainValidationAndTest(dataset):
    train_set = np.asarray(dataset[:-2000])
    validation_set = np.asarray(dataset[-2000:-1000])
    test_set = np.asarray(dataset[-1000:])
    # train_set,validation_set,test_set=senitizeUsingHashlib(train_set,validation_set,test_set)
    train_dataset, train_lables = splitImageAndLables(train_set)
    validation_dataset, validation_lables = splitImageAndLables(validation_set)
    test_dataset, test_lables = splitImageAndLables(test_set)
    return train_dataset, train_lables, validation_dataset, validation_lables, test_dataset, test_lables


def splitImageAndLables(data_set):
    images = []
    lables = []
    for tup in data_set:
        image = tup[0]
        lable = tup[1]
        try:
            resized_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            image_data = (resized_image - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            images.append(image_data)
        except IOError as e:
            print('Could not read:', image, ':', e, '- it\'s ok, skipping.')

        lables.append(lable)
    return np.asarray(images), np.asarray(lables)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    # print permutation
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def pickleProcess(dataset):
    random.shuffle(dataset)

    train_dataset, train_lables, validation_dataset, validation_lables, test_dataset, test_lables = splitIntoTrainValidationAndTest(
        dataset)
    train_dataset, train_lables = randomize(train_dataset, train_lables)
    validation_dataset, validation_lables = randomize(validation_dataset, validation_lables)
    test_dataset, test_lables = randomize(test_dataset, test_lables)
    print('Training:', train_dataset.shape, train_lables.shape)
    print('Validation:', validation_dataset.shape, validation_lables.shape)
    print('Testing:', test_dataset.shape, test_lables.shape)
    pickle_file = os.path.join(data_root, 'retail.pickle')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_lables,
            'valid_dataset': validation_dataset,
            'valid_labels': validation_lables,
            'test_dataset': test_dataset,
            'test_labels': test_lables,
        }
        cPickle.dump(save, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


# def senitizeUsingHashlib(train_set,validation_set,test_set):
#
#     train_hashes = [(hashlib.sha1(x).digest(),y) for x,y in train_set]
#     valid_hashes = [(hashlib.sha1(x).digest(),y) for x,y in validation_set]
#     test_hashes = [(hashlib.sha1(x).digest(),y) for x,y in test_set]
#
#     unique_train_dataset = set(train_hashes)
#     unique_valid_dataset = set(valid_hashes)
#     unique_test_dataset = set(test_hashes)
#
#     train_valid_overlap_hash = unique_train_dataset.intersection(set(valid_hashes))
#     train_test_overlap_hash = unique_train_dataset.intersection(set(test_hashes))
#     valid_test_overlap_hash = unique_valid_dataset.intersection(set(test_hashes))
#
#     final_train=unique_train_dataset.difference(train_valid_overlap_hash).difference(train_test_overlap_hash)
#     final_validation=unique_valid_dataset
#     final_test=unique_test_dataset.difference(valid_test_overlap_hash)
#
#     print('Duplicates between train and validation: ', len(train_valid_overlap_hash))
#     print('Duplicates between train and test: ', len(train_test_overlap_hash))
#     print('Duplicates between validation and test: ', len(valid_test_overlap_hash))
#     return final_train,final_validation,final_test


with open("dataset/mixture.pkl", "rb") as ds:
    images_with_lables = cPickle.load(ds)
    pickleProcess(images_with_lables)
