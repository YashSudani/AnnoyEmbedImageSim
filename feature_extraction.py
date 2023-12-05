import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class GetFeatureVector:
    """
    Get a feature vector using tensorflow classification model
    """

    def __init__(self):
        self.model = None
        self.base_model = None
        self.input_shape = (224, 224, 3)

    def load_model(self):
        """
        Load the model using tensorflow applications
        :return:
        """
        # Load the VGG19 model without including the top classification layers
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

    def prepare_model(self):
        """
        add identity layer to model
        :return:
        """
        # Add an identity layer (or any layer you want to replace the top layers)
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        feature_vector = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        # Create a new model with the modified top
        self.model = Model(inputs=self.base_model.input, outputs=feature_vector)

    def image_pre_process(self, image_path:str) -> np.array:
        """
        image pre processing for model input
        :param image_path:  image path
        :return: pre processed image
        """
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def get_feature_vector(self, image_path) -> np.array:
        if self.model is not None:
            img_arr = self.image_pre_process(image_path)
            return self.model.predict(img_arr)
        else:
            return "Please load the model"
