from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import PIL
import time
import hashlib

image_size = 640    # Pixel width and height.
channel_size = 3    # number of channels per pixel

train_size = 10
valid_size = 10
test_size = 10

def load_class(folder, min_num_images):
    """Load the data for a single class label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size, channel_size),
                            dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
        # Maybe Don't normalize?!
            #image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            image_data = ndimage.imread(image_file).astype(float)
            if image_data.shape != (image_size, image_size, channel_size):
                print('Unexpected image shape: %s' % str(image_data.shape))
                continue
            dataset[num_images, :, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        
    dataset = dataset[0:num_images, :, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))
        
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    """Loads the data for each class and pickle them as class.pickle"""
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_class(folder, min_num_images_per_class)
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)
    
    return dataset_names

def make_arrays(nb_rows, img_size, channel_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size, channel_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    """Splits the training data to Train and Validation datasets"""
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size, channel_size)
    train_dataset, train_labels = make_arrays(train_size, image_size, channel_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
        
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):       
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                                
                train_letter = letter_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
        
    return valid_dataset, valid_labels, train_dataset, train_labels
                       

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def RemoveDuplicates(src_pickle_file, data_root):
    with open(src_pickle_file, 'rb') as datafile:
        dsDic = pickle.load(datafile)
        
    train_hashes = [hashlib.sha1(x).digest() for x in dsDic['train_dataset']]
    test_hashes = [hashlib.sha1(x).digest() for x in dsDic['test_dataset']]
    valid_hashes = [hashlib.sha1(x).digest() for x in dsDic['valid_dataset']]

    print("{}|{}".format(len(test_hashes),len(dsDic['test_dataset'])))

        
    testInValid = np.in1d(test_hashes, valid_hashes)
    testInTrain = np.in1d(test_hashes,train_hashes)
    validInTrain = np.in1d(valid_hashes,train_hashes)

    valid_data_clean = dsDic['valid_dataset'][~validInTrain]
    valid_label_clean = dsDic['valid_labels'][~validInTrain]

    test_data_clean = dsDic['test_dataset'][~(testInTrain | testInValid)] 
    test_label_clean = dsDic['test_labels'][~(testInTrain | testInValid)]

    print ('{} validation samples were cleaned'.format(dsDic['valid_dataset'].shape[0]-valid_data_clean.shape[0]))
    print ('{} test samples were cleaned'.format(dsDic['test_dataset'].shape[0]-test_data_clean.shape[0]))

    dsDic['test_dataset'] = test_data_clean
    dsDic['test_labels'] = test_label_clean

    dsDic['valid_dataset'] = valid_data_clean
    dsDic['valid_labels'] = valid_label_clean

    print(list(dsDic.keys()))
    for key in dsDic.keys():
        print('{}:{}'.format(key, dsDic[key].shape, endline='--'))

    pickle_fileW = os.path.join(data_root, 'notMINSTUniq.pickle')

    with open(pickle_fileW, 'wb') as f:
        pickle.dump(dsDic, f, pickle.HIGHEST_PROTOCOL)
    

def PreprocessImageSet(train_dataset_root, test_dataset_root, out_folder):
    print('{} {}'.format(train_dataset_root, test_dataset_root))
    train_folders = [os.path.join(train_dataset_root, d) for d in sorted(os.listdir(train_dataset_root)) if os.path.isdir(os.path.join(train_dataset_root, d))]
    test_folders = [os.path.join(test_dataset_root, d) for d in sorted(os.listdir(test_dataset_root)) if os.path.isdir(os.path.join(test_dataset_root, d))]
    train_datasets = maybe_pickle(train_folders, 10)
    test_datasets = maybe_pickle(test_folders, 10)

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    pickle_file = os.path.join(out_folder, 'imageSet.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

def main():
    training_folder = "./train" 
    training_folder = input("Enter the path to training dataset: { default: ./train } ")

    test_folder = "./test" 
    test_folder = input("Enter the path to test dataset: { default: ./test } ")

    #Check the test and training paths are different

    out_folder = "." 
    out_folder = input("Enter output dir: { default: . }")
    print('{} {} {}'.format(training_folder, test_folder, out_folder))
    PreprocessImageSet(training_folder, test_folder, out_folder)

if __name__=='__main__':
    main()