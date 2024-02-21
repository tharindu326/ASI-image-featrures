import numpy as np
import utils.utilsgeo as ug
from sklearn.preprocessing import StandardScaler
from config import PATH
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
import copy
import os
from tqdm import tqdm
from transformers import ViTFeatureExtractor, TFAutoModel
import random
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

os.environ['PYTHONHASHSEED']=str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class ViTFeatureExtractorModel:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        # Initialize the feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        # Load the ViT model
        self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)

    def extract_features(self, image_paths):
        features = []
        for img_path in tqdm(image_paths, desc="Processing Images: ViT"):
            # Load the image
            img = load_img(img_path)
            img_array = img_to_array(img)
            # Perform necessary preprocessing for ViT
            inputs = self.feature_extractor(images=img_array, return_tensors="tf")
            # Extract features
            outputs = self.model(inputs['pixel_values'])
            # Extract the last hidden states
            feature = outputs.last_hidden_state[:, 0, :].numpy()
            features.append(feature)
        return np.array(features).squeeze()


class ResNet101FeatureExtractorModel:
    def __init__(self):
        # Initialize the feature extractor
        # Load ResNet101 model for feature extraction
        self.resnet_model = ResNet101(weights='imagenet', include_top=False, pooling='avg')
        
    # Image loading and feature extraction
    def extract_features(self, image_paths):
        features = []
        for img_path in tqdm(image_paths, desc="Processing Images : ResNet101"):
            # Load the image file, resizing it to 224x224 pixels (required by ResNet101 model)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            # Expand dimensions to fit model input format
            img_array_expanded = np.expand_dims(img_array, axis=0)
            # Preprocess the image data
            img_preprocessed = preprocess_input(img_array_expanded)
            # Extract features
            feature = self.resnet_model.predict(img_preprocessed)[0]
            features.append(feature)
        return np.array(features)
    
    
# class VGGFeatureExtractorModel:
#     def __init__(self):
#         # Initialize the feature extractor
#         # Load VGG16 model for feature extraction
#         self.vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
#      # Image loading and feature extraction
#     def extract_features(self, image_paths):
#         features = []
#         for img_path in tqdm(image_paths, desc="Processing Images : VGG"):
#             # Load the image file, resizing it to 224x224 pixels (required by VGG16 model)
#             img = load_img(img_path, target_size=(224, 224))
#             img_array = img_to_array(img)
#             # Expand dimensions to fit model input format
#             img_array_expanded = np.expand_dims(img_array, axis=0)
#             # Preprocess the image data
#             img_preprocessed = preprocess_input(img_array_expanded)
#             # Extract features
#             feature = self.vgg_model.predict(img_preprocessed)[0]
#             features.append(feature)
#         return np.array(features)
    
    
class VGGFeatureExtractorModel:
    def __init__(self, output_size):
        # Load VGG16 model without the top layer and with pooling set to None
        base_model = VGG16(weights='imagenet', include_top=False, pooling=None)
        # Add GlobalAveragePooling2D layer to reduce dimensions
        x = GlobalAveragePooling2D()(base_model.output)
        # Flatten the output to make it suitable for the Dense layer
        x = Flatten()(x)
        # Add a Dense layer with the desired output size
        x = Dense(output_size, activation='relu')(x)
        # Create the new model
        self.vgg_model = Model(inputs=base_model.input, outputs=x)

    def extract_features(self, image_paths):
        features = []
        for img_path in tqdm(image_paths, desc="Processing Images: VGG"):
            # Load and preprocess the image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array_expanded = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array_expanded)
            # Extract features
            feature = self.vgg_model.predict(img_preprocessed)[0]
            features.append(feature)
        return np.array(features)
        
        
class Geds:
    """

    """

    def __init__(self, id_dataset: str, num_nearest: int, geo: bool = True, euclidean: bool = True,
                 sequence: str = '', scale: bool = True, input_target_context=True,
                 input_dist_context_geo=True, input_dist_context_eucl=False, scale_euclidean=True,
                 scale_geo=False, image_features=True, image_feature_extractor="VGG", image_scale = True, num_image_features=None):
        """

        :param id_dataset:
        :param num_nearest:
        :param geo:
        :param euclidean:
        :param sequence:
        :param scale:
        :param input_target_context:
        :param input_dist_context_geo:
        :param input_dist_context_eucl:
        :param scale_euclidean:
        :param scale_geo:
        """

        self.image_features = image_features
        self.id_dataset = id_dataset
        self.scale = scale
        self.image_scale = image_scale
        self.num_nearest = num_nearest
        self.geo = geo
        self.euclidean = euclidean
        self.sequence = sequence
        self.input_target_context = input_target_context
        self.input_dist_context_geo = input_dist_context_geo
        self.input_dist_context_eucl = input_dist_context_eucl
        self.scale_euclidean = scale_euclidean
        self.scale_geo = scale_geo
        self.num_image_features = num_image_features
        if image_feature_extractor.lower() == "vgg":
            print(f"===============USING VGG FOR IMAGE FEATURE EXTRACTION ===============")
            self.image_feature_extractor = VGGFeatureExtractorModel(self.num_image_features)
        elif image_feature_extractor.lower() == "vit":
            self.image_feature_extractor = ViTFeatureExtractorModel()
            print(f"===============USING ViT FOR IMAGE FEATURE EXTRACTION ===============")
        elif image_feature_extractor.lower() == "resnet101":
            self.image_feature_extractor = ResNet101FeatureExtractorModel()
            print(f"===============USING ResNet101 FOR IMAGE FEATURE EXTRACTION ===============")

    
    def __call__(self):

        assert isinstance(self.id_dataset, object)
        base_dir = PATH + '/datasets/' + self.id_dataset
        train_image_data = f'{base_dir}/X_train_image.npy'
        test_image_data = f'{base_dir}/X_test_image.npy'
        X_train_image, X_test_image = None, None
        if self.image_features:
            print("================= USING IMAGE FEATURES =================")
            if not os.path.isfile(train_image_data):
                images_path = base_dir + '/map_images/'
                image_paths = []
                for file in os.listdir(images_path):
                    image_paths.append(f'{images_path}{file}')
                sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).replace("img", "").replace(".png", "")))

                # Splitting sorted_image_paths into training and testing based on the index
                if self.id_dataset == 'kc':
                    train_image_paths = sorted_image_paths[:17286]  # 0-17286 for training
                    test_image_paths = sorted_image_paths[17286:]
                elif self.id_dataset == 'poa':
                    train_image_paths = sorted_image_paths[:12294]  # 0-17286 for training
                    test_image_paths = sorted_image_paths[12294:]
                elif self.id_dataset == 'sp':
                    train_image_paths = sorted_image_paths[:55078]  # 0-17286 for training
                    test_image_paths = sorted_image_paths[55078:]
                elif self.id_dataset == 'fc':
                    train_image_paths = sorted_image_paths[:66510]  # 0-17286 for training
                    test_image_paths = sorted_image_paths[66510:]
                print(f'train_size: {len(train_image_paths)}')
                print(f'test_size: {len(test_image_paths)}')
                if len(train_image_paths) != 0 and len(test_image_paths) != None:
                    # Extract features for training and test images
                    X_train_image = self.image_feature_extractor.extract_features(train_image_paths)
                    X_test_image = self.image_feature_extractor.extract_features(test_image_paths)
                    
                    if self.image_scale:
                        scaler_image = StandardScaler()
                        X_train_image = scaler_image.fit_transform(X_train_image)
                        X_test_image = scaler_image.transform(X_test_image)

                    np.save(train_image_data, X_train_image)
                    np.save(test_image_data, X_test_image)
            else:
                X_train_image = np.load(train_image_data, allow_pickle=True)
                print('X_train_image', X_train_image.shape)
                X_test_image = np.load(test_image_data, allow_pickle=True)
                print('X_test_image', X_test_image.shape)
        data = np.load(PATH + '/datasets/'+ self.id_dataset + '/data'+self.sequence+'.npz', allow_pickle=True)

        # original data
        X_train = data['X_train']
        print('X_train', X_train.shape)
        X_test = data['X_test']
        print('X_test', X_test.shape)
        
        y_train = data['y_train']
        y_test = data['y_test']
        

        if self.geo:

            # the sequence of the nearest points (geodesic distance)
            nearest_train = data['idx_geo'][:X_train.shape[0], :self.num_nearest]
            nearest_dist_train = data['dist_geo'][:X_train.shape[0], :self.num_nearest]
            nearest_test = data['idx_geo'][X_train.shape[0]:, :self.num_nearest]
            nearest_dist_test = data['dist_geo'][X_train.shape[0]:, :self.num_nearest]

        else:
            nearest_train = 0
            nearest_dist_train = 0
            nearest_test = 0
            nearest_dist_test = 0

        if self.euclidean:

            # the sequence of the nearest points (Euclidean distance employing
            # the geographic distance for the closest points )
            nearest_train_eucli = data['idx_eucli'][:X_train.shape[0], :self.num_nearest]
            nearest_dist_train_eucli = data['dist_eucli'][:X_train.shape[0], :self.num_nearest]
            nearest_test_eucli = data['idx_eucli'][X_train.shape[0]:, :self.num_nearest]
            nearest_dist_test_eucli = data['dist_eucli'][X_train.shape[0]:, :self.num_nearest]
        else:
            nearest_train_eucli = 0
            nearest_dist_train_eucli = 0
            nearest_test_eucli = 0
            nearest_dist_test_eucli = 0

        # Concatenate the data
        X_train_test = np.concatenate((X_train, X_test), axis=0)
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        y_train_scale = copy.deepcopy(y_train)
        y_test_scale = copy.deepcopy(y_test)

        # preprocessing dataset

        scale = self.scale

        dist_train = copy.deepcopy(nearest_dist_train)
        dist_test = copy.deepcopy(nearest_dist_test)
        dist_train_eucli = copy.deepcopy(nearest_dist_train_eucli)
        dist_test_eucli = copy.deepcopy(nearest_dist_test_eucli)

        # Scaler
        if scale:

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            if self.geo:
                nearest_dist_train = scaler.fit_transform(nearest_dist_train)
                nearest_dist_test = scaler.fit_transform(nearest_dist_test)
                if self.scale_geo:
                    dist_train = scaler.fit_transform(dist_train)
                    dist_test = scaler.fit_transform(dist_test)

            if self.euclidean:
                nearest_dist_train_eucli = scaler.fit_transform(nearest_dist_train_eucli)
                nearest_dist_test_eucli = scaler.fit_transform(nearest_dist_test_eucli)

                if self.scale_euclidean:
                    dist_train_eucli = scaler.fit_transform(dist_train_eucli)
                    dist_test_eucli = scaler.fit_transform(dist_test_eucli)


            # Scaled because the target is used in neural networks
            y_train_scale = scaler.fit_transform(y_train_scale.reshape(-1, 1))

        else:

            y_train_scale = y_train_scale.reshape(-1, 1)

        # Training and test
        if self.geo:
            # Recover original data and include the target in sequence
            if self.input_target_context:
                train_x = ug.recover_original_data(nearest_train, np.concatenate((X_train, y_train_scale), axis=1))
                test_x = ug.recover_original_data(nearest_test, np.concatenate((X_train, y_train_scale), axis=1))

            else:
                train_x = ug.recover_original_data(nearest_train, X_train)
                test_x = ug.recover_original_data(nearest_test, X_train)

            # Concatenate the distance to the sequence
            if self.input_dist_context_geo:
            # Geo
                train_x = np.concatenate((train_x, np.reshape(nearest_dist_train, (
                    nearest_dist_train.shape[0], nearest_dist_train.shape[1], 1))), axis=2)
                test_x = np.concatenate((test_x, np.reshape(nearest_dist_test, (
                    nearest_dist_test.shape[0], nearest_dist_test.shape[1], 1))), axis=2)


        else:
            train_x = 0
            test_x = 0

        if self.euclidean:
            # Recover original data and include the target in sequence
            if self.input_target_context:
                train_x_eucli = ug.recover_original_data(nearest_train_eucli,
                                                         np.concatenate((X_train, y_train_scale), axis=1))
                test_x_eucli = ug.recover_original_data(nearest_test_eucli,
                                                        np.concatenate((X_train, y_train_scale), axis=1))
            else:
                train_x_eucli = ug.recover_original_data(nearest_train_eucli, X_train)
                test_x_eucli = ug.recover_original_data(nearest_test_eucli, X_train)
            # Concatenate the distance to the sequence
            if self.input_dist_context_eucl:
                # Euclidean
                train_x_eucli = np.concatenate((train_x_eucli, np.reshape(nearest_dist_train_eucli, (
                    nearest_dist_train_eucli.shape[0], nearest_dist_train_eucli.shape[1], 1))), axis=2)
                test_x_eucli = np.concatenate((test_x_eucli, np.reshape(nearest_dist_test_eucli, (
                    nearest_dist_test_eucli.shape[0], nearest_dist_test_eucli.shape[1], 1))), axis=2)
        else:
            train_x_eucli = 0
            test_x_eucli = 0

        # A sequence with original data of the nearest points more target (m, seq , features + target)
        if self.euclidean:
            context_struc_eucli_target_train = train_x_eucli[:, :, :]  # train_x_p
            context_struc_eucli_target_test = test_x_eucli[:, :, :]  # test_x_p
            # Distance to the nearest targets (euclidean) (m, seq)
            dist_eucli_train = dist_train_eucli  # train_x_e
            dist_eucli_test = dist_test_eucli  # test_x_e
        else:
            context_struc_eucli_target_train = 0
            context_struc_eucli_target_test = 0
            dist_eucli_train = 0
            dist_eucli_test = 0

        # Sequence with original data of the nearest phenomena more target and distance (m, seq , features + target +
        # dist)
        if self.geo:
            context_geo_target_dist_train = train_x[:, :, :]  # train_x_d
            context_geo_target_dist_test = test_x[:, :, :]  # test_x_d
            # Distance to the nearest targets (geodesica) (m, seq)
            dist_geo_train = dist_train  # train_x_g
            dist_geo_test = dist_test  # test_x_g


        else:
            context_geo_target_dist_train = 0
            context_geo_target_dist_test = 0
            dist_geo_train = 0
            dist_geo_test = 0

        # Original data are: X_train, X_test

        return context_struc_eucli_target_train, context_struc_eucli_target_test, \
               context_geo_target_dist_train, context_geo_target_dist_test, dist_geo_train, dist_geo_test,\
               dist_eucli_train, dist_eucli_test, X_train, X_test, y_train, y_test, y_train_scale, X_train_image, X_test_image