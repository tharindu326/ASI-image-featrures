import numpy as np
import os
from config import PATH
import asi.input_dataset as geds
import utils.utils as u
import utils.utilsdeep as ud
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Concatenate, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import LAMB
from asi.input_phenomenon import getinput
from asi.input_neighborhood import getcontext
from asi.interpolation import Interpolation as interpolation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
import random

os.environ['PYTHONHASHSEED']=str(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

class AttentionSpatialInterpolationModel:
    """

    """

    def __init__(self, id_dataset: str = None, num_nearest: int = None, early_stopping: bool = True,
                 geo: bool = True, euclidean: bool = True, scale: bool = True,
                 sequence: str = '', input_target_context=True, input_dist_context_geo=True,
                 input_dist_context_eucl=False, scale_euclidean=True, scale_geo=False, num_image_features=None, image_feature_extractor='VGG', image_scale=True):
        """

        :param id_dataset:
        :param num_nearest:
        :param early_stopping:
        :param geo:
        :param euclidean:
        :param scale:
        :param sequence:
        :param input_target_context:
        :param input_dist_context_geo:
        :param input_dist_context_eucl:
        :param scale_euclidean:
        :param scale_geo:
        """

        self.num_nearest = num_nearest
        self.early_stopping = early_stopping
        self.geo = geo
        self.euclidean = euclidean
        self.scale = scale
        self.sequence = sequence
        self.input_target_context = input_target_context
        self.input_dist_context_geo = input_dist_context_geo
        self.input_dist_context_eucl = input_dist_context_eucl
        self.scale_euclidean = scale_euclidean
        self.scale_geo = scale_geo
        # Location of the files
        self.path = PATH

        ######################## Dataset ########################

        # id dataset
        self.id_dataset = id_dataset
        if num_image_features is None:
            image_features = False
        else:
            image_features = True
            
        self.train_x_p, self.test_x_p, \
        self.train_x_d, self.test_x_d, \
        self.train_x_g, self.test_x_g, \
        self.train_x_e, self.test_x_e, \
        self.X_train, self.X_test, self.y_train,\
        self.y_test, self.y_train_scale, self.X_train_image, self.X_test_image = geds.Geds(id_dataset=self.id_dataset,
                                                     num_nearest=self.num_nearest,
                                                     geo=self.geo,
                                                     euclidean=self.euclidean, scale=self.scale,
                                                     sequence=self.sequence,
                                                     input_target_context=self.input_target_context,
                                                     input_dist_context_geo=self.input_dist_context_geo,
                                                     input_dist_context_eucl=self.input_dist_context_eucl,
                                                     scale_euclidean=self.scale_euclidean,
                                                     scale_geo=self.scale_geo, image_features=image_features, 
                                                     image_feature_extractor=image_feature_extractor,
                                                     image_scale = image_scale)()
        
        # the shape of the features datasets
        # input
        # the features of the points (phenomenon)
        self.shape_input_phe = self.X_train.shape[1]
        # the features of the points (phenomenon) without lat and long
        self.shape_input_phe_w_lat_long = self.X_train[:, 2:].shape[1]

        # context
        # (m, seq , features + target + dist)
        if self.geo:
            self.shape_context_geo_target_dist = self.train_x_d.shape[2]
        else:
            self.shape_context_geo_target_dist = 0

        # (m, seq , features + target) the features of the context points (phenomenon) without lat, long and target
        self.shape_context_struc_w_lat_long = self.X_train[:, 2:].shape[1]

        # (m, seq , features + target)
        if self.euclidean:
            self.shape_context_struc_target = self.train_x_p.shape[2]
        else:
            self.shape_context_struc_target = 0

        ############## check id_dataset directory exists ######################

        # logs
        if not os.path.exists(self.path + '/logs/hyperparameters/' + self.id_dataset):
            os.makedirs(self.path + '/logs/hyperparameters/' + self.id_dataset)
        if not os.path.exists(self.path + '/logs/models/' + self.id_dataset):
            os.makedirs(self.path + '/logs/models/' + self.id_dataset)

        # general outputs
        if not os.path.exists(self.path + '/output/hyperparameters/' + self.id_dataset):
            os.makedirs(self.path + '/output/hyperparameters/' + self.id_dataset)
        if not os.path.exists(self.path + '/output/images/architecture/' + self.id_dataset):
            os.makedirs(self.path + '/output/images/architecture/' + self.id_dataset)
        if not os.path.exists(self.path + '/output/images/nearest/' + self.id_dataset):
            os.makedirs(self.path + '/output/images/nearest/' + self.id_dataset)
        if not os.path.exists(self.path + '/output/models/' + self.id_dataset):
            os.makedirs(self.path + '/output/models/' + self.id_dataset)
        if not os.path.exists(self.path + '/output/result/' + self.id_dataset):
            os.makedirs(self.path + '/output/result/' + self.id_dataset)

        # notebooks
        if not os.path.exists(self.path + '/notebooks/' + self.id_dataset):
            os.makedirs(self.path + '/notebooks/' + self.id_dataset)

    # Clear clutter from previous Keras session graphs.
    clear_session()

    def build(self, geointerpolation: str = 'simple asi', sigma=None, optimizer: str = 'adam',
              type_compat_funct_eucli: str = 'identity',
              type_compat_funct_geo: str = 'kernel_gaussiano', activation:str = 'elu',
              num_features_extras_struct: int = 1, num_features_extras_geo: int = 2, cal_dist_struct: bool = False,
              cal_dist_geo: bool = False, learning_rate: float = None, num_layers: int = None,
              num_neuron: int = None, size_embedded: int = None, graph_label: str = None,
              num_nearest_geo: int = 20, num_nearest_eucli: int = 20, num_image_features: int = None):
        """

        :param num_nearest_eucli:
        :param num_nearest_geo:
        :param type_compat_funct_eucli: compatibility function used in structured attention
        :param geointerpolation:
        :param sigma: adjustment factor of the Gaussian curve
        :param optimizer:
        :param type_compat_funct_geo: compatibility function used in geo attention
        :param num_features_extras_struct: an extra feature, in addition to the target, to be added in structured attention
        :param num_features_extras_geo: an extra feature, in addition to the target, to be added in geo attention
        :param cal_dist_struct:
        :param cal_dist_geo:
        :param learning_rate:
        :param num_layers:
        :param num_neuron:
        :param size_embedded:
        :param graph_label: name of the softmax layer
        :return:
        """

        if sigma is None:
            sigma = [0, 0]

        # self.optimizer = Adam(learning_rate=learning_rate)
        # Define an exponential decay learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        # Use this learning rate schedule in the Adam optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=lr_schedule)
        elif optimizer == 'lamb':
            self.optimizer=LAMB(learning_rate=lr_schedule)
        print(f"============= using {optimizer} optimizer =============")

        # create model
        ######################## Layer input of the points ########################

        """
        input_phenomenon: phenomena input features
        input_phe_w_lat_long: input features of the phenomenon without lat and long
        repeat_phe_w_lat_long: repeats the structural phenomenon data according to 
        the number of closest neighbors. Not part of the input
        """

        input_phenomenon, \
        input_phe_w_lat_long, \
        repeat_phe_w_lat_long = getinput(self.shape_input_phe, self.shape_input_phe_w_lat_long, self.num_nearest)

        # New input for image features
        if num_image_features is not None:
            image_input = Input(shape=(num_image_features,), name='image_input')
        
        ######################## Layer input of the sequences ########################

        """
        context_geo_target_dist: original data from the nearest points plus target and distance (m, seq , features + target + dist)
        context_struc_w_lat_long: original data from the nearest points less lat and long (m, seq , features - 2)
        context_struc_[]_target: original data from the nearest points plus target (m, seq , features + target)
        dist_geo: geodesic distance from nearest points (m, seq)
        dist_eucli: euclidean distance from nearest points (m, seq)
        dist_cosine: cosine distance from nearest points (m, seq)
        """

        context_geo_target_dist, \
        context_struc_w_lat_long, \
        context_struc_eucli_target, \
        dist_geo, \
        dist_eucli = getcontext(self.shape_context_struc_target,
                                self.shape_context_struc_w_lat_long,
                                self.shape_context_geo_target_dist, self.num_nearest,
                                self.geo, self.euclidean, num_nearest_geo, num_nearest_eucli)

        ######################## Interpolation ########################

        embbeding = interpolation(
            geointerpolation=geointerpolation,
            shape_input_phenomenon=self.shape_input_phe,
            shape_input_phenomenon_eucl=self.shape_input_phe_w_lat_long,
            input_phenomenon=input_phenomenon,
            context_struc_eucli_target=context_struc_eucli_target,
            context_geo_target_dist=context_geo_target_dist,
            type_compat_funct_eucli=type_compat_funct_eucli,
            type_compat_funct_geo=type_compat_funct_geo,
            num_features_extras_struct=num_features_extras_struct,
            num_features_extras_geo=num_features_extras_geo,
            cal_dist_struct=cal_dist_struct,
            cal_dist_geo=cal_dist_geo,
            graph_label=graph_label,
            dist_eucli=dist_eucli,
            dist_geo=dist_geo,
            sigma=sigma,
            num_nearest=self.num_nearest,
            num_nearest_geo=num_nearest_geo,
            num_nearest_eucli=num_nearest_eucli,
            num_neuron=num_neuron,
            num_layers=num_layers,
            size_embedded=size_embedded,
            input_phe_w_lat_long=input_phe_w_lat_long,
            geo=self.geo,
            euclidean=self.euclidean,
            activation=activation

        ).run()
        
        ######################## Regression Layer ########################

        if num_image_features is not None:
            concatenated_features = Concatenate()([embbeding, image_input])
            # Continue with the concatenated data through dense layers or directly to output
            x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concatenated_features)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = Dropout(0.5)(x)
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = Dropout(0.5)(x)
            main_output = Dense(1, activation='linear', name='main_output')(x)
        else:
            # x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(embbeding)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = Dropout(0.5)(x)
            # x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = Dropout(0.5)(x)
            # main_output = Dense(1, activation='linear', name='main_output')(x)
            main_output = Dense(1, activation='linear', name='main_output')(embbeding)
        
        # if num_image_features is not None:
        #     concatenated_features = Concatenate()([embbeding, image_input])
        #     # Continue with the concatenated data through dense layers or directly to output
        #     x = Dense(128, activation='relu')(concatenated_features)
        #     x = Dense(64, activation='relu')(x)
        #     main_output = Dense(1, activation='linear', name='main_output')(x)
        # else:
        #     main_output = Dense(1, activation='linear', name='main_output')(embbeding)
            
        ######################## model ########################

        # Link the graph
        if num_image_features is not None:
            if self.geo and self.euclidean:
                model = Model(inputs=[input_phenomenon, context_geo_target_dist,
                                    context_struc_eucli_target,
                                    dist_geo, dist_eucli, image_input], outputs=[main_output])
            elif self.geo:
                model = Model(inputs=[input_phenomenon, context_geo_target_dist,
                                    dist_geo, image_input], outputs=[main_output])
            elif self.euclidean:
                model = Model(inputs=[input_phenomenon,
                                    context_struc_eucli_target,
                                    dist_eucli, image_input], outputs=[main_output])
            else:
                model = Model(inputs=[input_phenomenon, image_input], outputs=[main_output])
        else:
            if self.geo and self.euclidean:
                model = Model(inputs=[input_phenomenon, context_geo_target_dist,
                                    context_struc_eucli_target,
                                    dist_geo, dist_eucli], outputs=[main_output])
            elif self.geo:
                model = Model(inputs=[input_phenomenon, context_geo_target_dist,
                                    dist_geo], outputs=[main_output])
            elif self.euclidean:
                model = Model(inputs=[input_phenomenon,
                                    context_struc_eucli_target,
                                    dist_eucli], outputs=[main_output])
            else:
                model = Model(inputs=[input_phenomenon], outputs=[main_output])

        # Compile the model
        model.compile(optimizer=self.optimizer, loss='mae', metrics=[RootMeanSquaredError()])

        return model

    def train(self, model, num_nearest_geo, num_nearest_eucli,
              epochs: int = None, batch_size: int = None,
              validation_split: float = None, label: str = None):

        """

        :param model: The model to be trained.
        :param image_features: The numpy array containing extracted features from images.
        :param num_nearest_geo: Number of nearest geospatial points considered.
        :param num_nearest_eucli: Number of nearest Euclidean points considered.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size for training.
        :param validation_split: Fraction of data to use for validation.
        :param label: Label for the training session, used in naming checkpoint files.
        :return: The location of the saved weights and the fit history.
        """

        # checkpoint
        weights_locate = label + '_weights.hdf5'
        filepath = self.path + '/output/models/' + self.id_dataset + '/' + weights_locate
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min')

        # Fit to model
        
        if self.X_train_image is not None:
            if self.geo and self.euclidean:
                #
                features = [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                            self.train_x_p[:, :num_nearest_eucli, :], self.train_x_g[:, :num_nearest_geo],
                            self.train_x_e[:, :num_nearest_eucli], self.X_train_image]

            elif self.geo and not self.euclidean:
                #
                features = [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                            self.train_x_g[:, :num_nearest_geo], self.X_train_image]

            elif self.euclidean and not self.geo:
                #
                features = [self.X_train[:, :], self.train_x_p[:, :num_nearest_eucli, :],
                            self.train_x_e[:, :num_nearest_eucli], self.X_train_image]

            elif not self.geo and not self.euclidean:
                #
                features = [self.X_train[:, :], self.X_train_image]
        else:
            if self.geo and self.euclidean:
                #
                features = [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                            self.train_x_p[:, :num_nearest_eucli, :], self.train_x_g[:, :num_nearest_geo],
                            self.train_x_e[:, :num_nearest_eucli]]

            elif self.geo and not self.euclidean:
                #
                features = [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                            self.train_x_g[:, :num_nearest_geo]]

            elif self.euclidean and not self.geo:
                #
                features = [self.X_train[:, :], self.train_x_p[:, :num_nearest_eucli, :],
                            self.train_x_e[:, :num_nearest_eucli]]

            elif not self.geo and not self.euclidean:
                #
                features = [self.X_train[:, :]]

        if self.early_stopping:
            fit = model.fit(features, [self.y_train], epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split, verbose=1,
                            callbacks=[ud.history, ud.early_stopping, checkpoint])

        else:
            fit = model.fit(features, [self.y_train], epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split, verbose=1, callbacks=[ud.history, checkpoint])

        return weights_locate, fit

    def predict_value(self, model, weights, num_nearest_geo, num_nearest_eucli, scale_log: bool = True,
                      batch_size: int = None):

        """

        :param model:
        :param weights:
        :param num_nearest_geo:
        :param num_nearest_eucli:
        :param scale_log:
        :param batch_size:
        :return:
        """

        # Predict

        # try:

        # load weights
        print(weights)
        print("==========================================", self.path + '/output/models/' + self.id_dataset + '/' + weights)
        model.load_weights(self.path + '/output/models/' + self.id_dataset + '/' + weights)

        # model compile
        model.compile(optimizer=self.optimizer, loss='mae')

        # predict the value
        if self.X_test_image is not None and self.X_train_image is not None:
            if self.geo and self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :], self.test_x_d[:, :num_nearest_geo, :],
                    self.test_x_p[:, :num_nearest_eucli, :], self.test_x_g[:, :num_nearest_geo],
                    self.test_x_e[:, :num_nearest_eucli], self.X_test_image], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                    self.train_x_p[:, :num_nearest_eucli, :], self.train_x_g[:, :num_nearest_geo],
                    self.train_x_e[:, :num_nearest_eucli], self.X_train_image], batch_size=batch_size)

            elif self.geo:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :], self.test_x_d[:, :num_nearest_geo, :],
                    self.test_x_g[:, :num_nearest_geo], self.X_test_image],
                    batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                    self.train_x_g[:, :num_nearest_geo], self.X_train_image],
                    batch_size=batch_size)
            elif self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :],
                    self.test_x_p[:, :num_nearest_eucli, :], self.test_x_e[:, :num_nearest_eucli], self.X_test_image], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :],
                    self.train_x_p[:, :num_nearest_eucli, :], self.train_x_e[:, :num_nearest_eucli], self.X_train_image],
                    batch_size=batch_size)
            elif not self.geo and not self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :],self.X_test_image], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :], self.X_train_image],
                    batch_size=batch_size)
        else:
            if self.geo and self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :], self.test_x_d[:, :num_nearest_geo, :],
                    self.test_x_p[:, :num_nearest_eucli, :], self.test_x_g[:, :num_nearest_geo],
                    self.test_x_e[:, :num_nearest_eucli]], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                    self.train_x_p[:, :num_nearest_eucli, :], self.train_x_g[:, :num_nearest_geo],
                    self.train_x_e[:, :num_nearest_eucli]], batch_size=batch_size)

            elif self.geo:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :], self.test_x_d[:, :num_nearest_geo, :],
                    self.test_x_g[:, :num_nearest_geo]],
                    batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :], self.train_x_d[:, :num_nearest_geo, :],
                    self.train_x_g[:, :num_nearest_geo]],
                    batch_size=batch_size)
            elif self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :],
                    self.test_x_p[:, :num_nearest_eucli, :], self.test_x_e[:, :num_nearest_eucli]], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :],
                    self.train_x_p[:, :num_nearest_eucli, :], self.train_x_e[:, :num_nearest_eucli]],
                    batch_size=batch_size)
            elif not self.geo and not self.euclidean:
                # test
                predictions_test = model.predict(
                    [self.X_test[:, :]], batch_size=batch_size)
                # train
                predictions_train = model.predict(
                    [self.X_train[:, :]],
                    batch_size=batch_size)

        predictions_train_dim = np.reshape(predictions_train, (self.y_train.shape[0],))
        predictions_test_dim = np.reshape(predictions_test, (self.y_test.shape[0],))

        if scale_log:
            ############################################ test ############################################
            rmse_test = mean_squared_error(np.exp(self.y_test), np.exp(predictions_test_dim))

            mae_log_test = mean_absolute_error(self.y_test, predictions_test_dim)

            mape_test = u.mean_absolute_percentage_error(np.exp(self.y_test), np.exp(predictions_test_dim))

            ############################################ train ############################################
            rmse_train = mean_squared_error(np.exp(self.y_train), np.exp(predictions_train_dim))

            mae_log_train = mean_absolute_error(self.y_train, predictions_train_dim)

            mape_train = u.mean_absolute_percentage_error(np.exp(self.y_train), np.exp(predictions_train_dim))

        else:
            ############################################ test ############################################
            rmse_test = mean_squared_error(self.y_test, predictions_test_dim)

            mae_log_test = mean_absolute_error(self.y_test, predictions_test_dim)

            mape_test = u.mean_absolute_percentage_error(self.y_test, predictions_test_dim)

            ############################################ train ############################################
            rmse_train = mean_squared_error(self.y_train, predictions_train_dim)

            mae_log_train = mean_absolute_error(self.y_train, predictions_train_dim)

            mape_train = u.mean_absolute_percentage_error(self.y_train, predictions_train_dim)


        return (mae_log_test, np.sqrt(rmse_test), mape_test, mae_log_train, np.sqrt(rmse_train),  mape_train)
        # except:

        #     return (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

    def architecture(self, fitted_model, label: str = None):
        """

        :param fitted_model:
        :param label:
        :return:
        """
        # save image architecture model
        file_path_arc = self.path + '/output/images/architecture/' + self.id_dataset + '/' + label + '.png'
        plot_model(fitted_model, to_file=file_path_arc, show_shapes=True, show_layer_names=True)

    def output_layer(self, model, weight, layer, data, batch, file_name):
        """

        :param model:
        :param weight:
        :param layer:
        :param data:
        :param batch:
        :param file_name:
        :return:
        """

        # load weights
        model.load_weights(self.path + '/output/models/' + self.id_dataset + '/' + weight)

        # Compilando o modelo
        model.compile(optimizer=self.optimizer, loss='mae')

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer).output)

        predict = intermediate_layer_model.predict(data, batch_size=batch)

        np.save(self.path + '/output/result/' + self.id_dataset + '/'+file_name, predict)


        return predict
