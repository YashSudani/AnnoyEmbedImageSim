import os
import glob
from annoy import AnnoyIndex
from feature_extraction import GetFeatureVector


def train(image_folder_path: str):
    """
    extract feature from image and create annoy file
    :param image_folder_path:
    :return:
    """
    # building annoy indexer
    t = AnnoyIndex(2048, 'angular')

    folder_name = image_folder_path
    image_list = glob.glob(os.path.join(folder_name, "*"))
    image_list.sort()

    GetFeatureVector_obj = GetFeatureVector()
    GetFeatureVector_obj.load_model()
    GetFeatureVector_obj.prepare_model()

    for idx, image_path in enumerate(image_list):
        feature_vector = GetFeatureVector_obj.get_feature_vector(image_path=image_path)
        t.add_item(idx, feature_vector[0])

    # number of tree nodes
    t.build(10)
    if os.path.exists('cat_dog.ann'):
        os.remove('cat_dog.ann')
    t.save('cat_dog.ann')
    del GetFeatureVector_obj
