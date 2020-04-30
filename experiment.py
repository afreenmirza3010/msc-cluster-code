
from comet_ml import Experiment as ex
from deoxys.experiment import Experiment
from deoxys.utils import read_file
from deoxys.model import model_from_full_config
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from deoxys.loaders.architecture import BaseModelLoader
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Sequential
from tensorflow import image
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import tensorboard
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np



from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *



import numpy as np
import math

from scipy.signal import resample_poly

from deoxys.model.layers import layer_from_config
from deoxys.customize import custom_architecture
import os
import h5py

@custom_architecture
class Vnet(BaseModelLoader):
    """
    Create a unet neural network from layers

    :raises NotImplementedError: volumn adjustment in skip connection
    doesn't support for 3D unet
    """

    def resize_by_axis(self, img, dim_1, dim_2, ax):
        resized_list = []
        #print(img.shape, ax, dim_1, dim_2)
        unstack_img_depth_list = tf.unstack(img, axis=ax)
        for j in unstack_img_depth_list:
            resized_list.append(
                image.resize(j, [dim_1, dim_2], method='bilinear'))
        stack_img = tf.stack(resized_list, axis=ax)
        #print(stack_img.shape)
        return stack_img

    def resize_along_dim(self, img, new_dim):
        dim_1, dim_2, dim_3 = new_dim

        resized_along_depth = self.resize_by_axis(img, dim_1, dim_2, 3)
        resized_along_width = self.resize_by_axis(resized_along_depth, dim_1, dim_3, 2)
        return resized_along_width


    def load(self):
        """
        Load the unet neural network.
        Example of Configuration for `layers`:
        ```
        [
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_1",
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "MaxPooling2D"
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_2",
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            ...
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_5",
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "MaxPooling2D"
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 128,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 128,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_1",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 32,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_1",
                    "conv_5"
                ]
            },
            ...
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_4",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_4",
                    "conv_2"
                ]
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "name": "conv_T_5",
                "class_name": "Conv2DTranspose",
                "config": {
                    "filters": 2,
                    "kernel_size": 3,
                    "strides": [
                        2,
                        2
                    ],
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                },
                "inputs": [
                    "conv_T_5",
                    "conv_1"
                ]
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "filters": 1,
                    "kernel_size": 1,
                    "activation": "sigmoid"
                }
            }
        ]
        ```

        :raises NotImplementedError: volumn adjustment in skip connection
        doesn't support
        :return: A neural network with unet structure
        :rtype: tensorflow.keras.models.Model
        """
        global next_input
        layers = [Input(**self._input_params)]
        saved_input = {}

        for i, layer in enumerate(self._layers):
            next_tensor = layer_from_config(layer)

            if 'inputs' in layer:
                inputs = []
                size_factors = None
                for input_name in layer['inputs']:
                    if size_factors:
                        if size_factors == saved_input[
                                input_name].get_shape().as_list()[1:-1]:
                            next_input = saved_input[input_name]
                        else:
                            if len(size_factors) == 2:
                                next_input = image.resize(
                                    saved_input[input_name],
                                    size_factors,
                                    # preserve_aspect_ratio=True,
                                    method='bilinear')
                            elif len(size_factors) == 3:

                                next_input = self.resize_along_dim(
                                    saved_input[input_name],
                                    size_factors
                                )

                            else:
                                raise NotImplemented("Image shape is not supported ")
                        inputs.append(next_input)

                    else:
                        inputs.append(saved_input[input_name])
                        size_factors = saved_input[
                            input_name].get_shape().as_list()[1:-1]
                connected_input = concatenate(inputs)
            else:
                connected_input = layers[i]

            next_layer = next_tensor(connected_input)

            if 'normalizer' in layer:
                next_layer = layer_from_config(layer['normalizer'])(next_layer)

            if 'name' in layer:
                saved_input[layer['name']] = next_layer

            layers.append(next_layer)

        return KerasModel(inputs=layers[0], outputs=layers[-1])


if __name__ == '__main__':
    def make_mesh(image, threshold=-300, step_size=1):
        print ("Transposing surface")

        p = image.transpose(2, 1, 0)

        print("calculating surface")

        verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size,
                                                         allow_degenerate=True)
        return verts, faces


    def plotly_3d(verts, faces):
        x, y, z = zip(*verts)

        print("Drawing")


        # Make the colormap single color since the axes are positional not intensity.
        #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
        colormap = ['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']

        fig = FF.create_trisurf(x=x,
                                y=y,
                                z=z,
                                plot_edges=False,
                                colormap=colormap,
                                simplices=faces,
                                backgroundcolor='rgb(64, 64, 64)',
                                title="Interactive Visualization")
        iplot(fig)


    def plt_3d(verts, faces):
        print("drwaing")

        x, y, z = zip(*verts)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
        face_color = [1, 1, 0.9]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, max(x))
        ax.set_ylim(0, max(y))
        ax.set_zlim(0, max(z))
        ax.set_facecolor((0.7, 0.7, 0.7))
        plt.show()


    with h5py.File('/work/users/afmi_msc_thesis_delete_01oct2020/head_neck_new4.h5','r+') as hdf:
        print(hdf.keys())

        images = hdf['fold_10']['input'][2]
        print(images)

        new_image = images[:,:,:,0]


        print(new_image.shape)
        v, f = make_mesh(new_image, 350)
        plt_3d(v, f)
    #     masks = hdf['fold_19']['target'][2]
    #
    #     f, axes = plt.subplots(5, 5, figsize=(18, 18))
    #     ax_i = 0
    #     for i in range(25):
    #         ax = axes[ax_i // 5][ax_i % 5]
    #         ax.imshow(images[i,:,:,0], 'gray')
    #         ax.contour(masks[:,:], 1, colors='yellow')
    #         ax.set_title(f"Slice {i}")
    #         ax_i += 1
    #     plt.show()
    # #     #
        # plt.imshow(first_mask[23],aspect=0.5)
        # # struct_arr2 = first_vol.T
        # # plt.imshow(struct_arr2[76],aspect=0.5)
        # plt.colorbar()
        # plt.title('plot for images')
        # plt.show()




    #ex_comet = ex(api_key="zoPcSaPo6mhKthsM8SOcgq9Uk",project_name="masterthesisafreen", workspace="afreen3010")
    #
    # config = read_file('json/small_vnet_modified.json')
    # experiment = Experiment()
    #
    # experiment.from_full_config(config).run_experiment()
    # experiment.from_full_config(config).run_test()

    # with h5py.File('logs/test/prediction_test.h5','r+') as hf:
    #    print(hf['predicted'][4])


    # # # defining model here
    # model = model_from_full_config(config)
    # x, y = model.data_reader.train_generator.generate().__next__()
    #breakpoint()
    #with ex_comet.train():
        #x, y = model.data_reader.train_generator.generate().__next__()
        #model.fit(x, y)




    # model.fit(x, y)
