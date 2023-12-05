import os
import cv2
import glob
import numpy as np
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from feature_extraction import GetFeatureVector


def show_images(images: np.array, rows: int, cols: int, figsize=(10, 10)):
    """
    show image in tile
    :param images: image array containing images in numpy formate
    :param rows: number of row
    :param cols: number of column
    :param figsize: size of output figure
    :return:
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    ax = axes.flat
    for idx, img_np in enumerate(images):
        ax[idx].imshow(img_np)
        ax[idx].axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()


def inference(image_folder_path: str, input_img: str, num_similar_image: int):
    """
    find similar image
    :param image_folder_path: image folder path which is used during training
    :param input_img: input image which needs to compare
    :param num_similar_image: int number, this much of similar looking images will be displayed in output
    :return:
    """

    t = AnnoyIndex(2048, 'angular')

    folder_name = image_folder_path
    image_list = glob.glob(os.path.join(folder_name, "*"))
    image_list.sort()

    GetFeatureVector_obj = GetFeatureVector()
    GetFeatureVector_obj.load_model()
    GetFeatureVector_obj.prepare_model()

    input_img_path = input_img
    input_img = cv2.imread(input_img_path)
    input_img = cv2.resize(input_img, (224, 224))
    detected_image_list = [input_img]
    image_path_list = [input_img_path]
    feature_vector = GetFeatureVector_obj.get_feature_vector(image_path=input_img_path)
    t.load('cat_dog.ann')  # super-fast, will just mmap the file
    match_images_index = t.get_nns_by_vector(feature_vector[0], num_similar_image)  # will find the 1000 nearest neighbors
    for idx in match_images_index:
        image_path = image_list[idx]
        image_path_list.append(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        detected_image_list.append(img)

    # rows = 3  # Number of rows in the grid
    # cols = 4  # Number of columns in the grid
    # show_images(detected_image_list, rows, cols)
    del GetFeatureVector_obj
    return image_path_list
