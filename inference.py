from asi.model import AttentionSpatialInterpolationModel as asi
import json 

class train:

    def __init__(self, sigma, learning_rate, batch_size, num_neuron, num_layers, size_embedded,
                 num_nearest_geo, num_nearest_eucli, id_dataset, label, graph_label, num_nearest,
                 epochs, validation_split, early_stopping, optimier, **kwargs):

        """

        :param sigma:
        :param learning_rate:
        :param batch_size:
        :param num_neuron:
        :param num_layers:
        :param size_embedded:
        :param num_nearest_geo:
        :param num_nearest_eucli:
        :param id_dataset:
        :param label:
        :param graph_label:
        :param num_nearest:
        :param epochs:
        :param validation_split:
        :param early_stopping:
        :param optimier:
        :param kwargs:
        """

        self.NUM_NEAREST = num_nearest
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.NUM_NEURON = num_neuron
        self.NUM_LAYERS = num_layers
        self.SIZE_EMBEDDED = size_embedded
        self.NUM_NEAREST_GEO = num_nearest_geo
        self.NUM_NEAREST_EUCLI = num_nearest_eucli
        self.ID_DATASET = id_dataset
        self.EPOCHS = epochs
        self.OPTIMIZER = optimier
        self.VALIDATION_SPLIT = validation_split
        self.LABEL = label
        self.EARLY_STOPPING = early_stopping
        self.GRAPH_LABEL = graph_label
        self.num_image_features = kwargs.get('num_image_features')
        self.scale = kwargs.get('scale', False)
        self.image_scale = kwargs.get('image_scale', True)
        self.image_feature_extractor = kwargs.get('image_feature_extractor', 'VGG')
        self.weight = kwargs.get('weight', 'asi_kc_weights_240216_109K.hdf5')


    def __call__(self):

        ####################################### Model ##########################################

        # build of the object
        spatial = asi(id_dataset=self.ID_DATASET,
                      num_nearest=self.NUM_NEAREST,
                      early_stopping=self.EARLY_STOPPING,
                      num_image_features=self.num_image_features, scale=self.scale, 
                      image_feature_extractor=self.image_feature_extractor, image_scale=self.image_scale)

        # build of the model
        model = spatial.build(sigma=[0, self.SIGMA],
                              optimizer=self.OPTIMIZER,
                              learning_rate=self.LEARNING_RATE,
                              num_layers=self.NUM_LAYERS,
                              num_neuron=self.NUM_NEURON,
                              size_embedded=self.SIZE_EMBEDDED,
                              graph_label=self.GRAPH_LABEL,
                              num_nearest_geo=self.NUM_NEAREST_GEO,
                              num_nearest_eucli=self.NUM_NEAREST_EUCLI,
                              num_image_features=self.num_image_features)

        # prediction
        result = spatial.predict_value(model=model,
                                       weights=self.weight,
                                       num_nearest_geo=self.NUM_NEAREST_GEO,
                                       num_nearest_eucli=self.NUM_NEAREST_EUCLI)


        return result
    
    
if __name__ == "__main__":
    # %matplotlib inline
    # import sys
    # sys.path.append("../../")

    from matplotlib import rcParams
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    
    hyperparameter={
                        "num_nearest":60,
                        "sigma":2,
                        "learning_rate":0.001,
                        "batch_size":250,
                        "num_neuron":60,
                        "num_layers":3,
                        "size_embedded":50,
                        "num_nearest_geo":45,
                        "num_nearest_eucli":45,
                        "id_dataset":'kc',
                        "epochs":450,
                        "optimier":'adam',
                        "validation_split":0.1,
                        "label":'asi_poa',
                        "early_stopping": False,
                        "graph_label":'matrix',
                        "num_image_features": 512,
                        "scale": True,
                        "image_feature_extractor": 'vgg', # 'vgg'; num_image_features-512 , 'vit'; num_image_features-768, 'resnet101'; num_image_features-2048
                        "image_scale": True, 
                        "weight": 'asi_kc_weights_240216_109K.hdf5'  # weight need to be in the path accordingly. ex; ASI/output/models/kc/
                        }
    
    spatial = train(**hyperparameter)
    result = spatial()
    mae_test, rmse_test, mape_test, mae_train, rmse_train,  mape_train = result
    res = {
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'mape_test': mape_test,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'mape_train': mape_train
    }
    print(json.dumps(res, indent=4))