
from comet_ml import Experiment as ex
from deoxys.experiment import Experiment
from deoxys.utils import read_file
from deoxys.model import model_from_full_config
import matplotlib.pyplot as plt


from deoxys.loaders.architecture import BaseModelLoader
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Sequential
from tensorflow import image
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import tensorboard
import numpy as np
import scipy.ndimage
import math
from deoxys.model.layers import layer_from_config
from deoxys.customize import custom_architecture
import os
import h5py




if __name__ == '__main__':
    

    #ex_comet = ex(api_key="zoPcSaPo6mhKthsM8SOcgq9Uk",project_name="masterthesisafreen", workspace="afreen3010")
    #
    config = read_file('json/layer_32_filter_new.json')
    experiment = Experiment()
    #
    #from pdb import set_trace; set_trace()
    experiment.from_full_config(config).run_experiment(train_history_log=True,
        model_checkpoint_period=10,epochs=100).plot_performance()
    

    
