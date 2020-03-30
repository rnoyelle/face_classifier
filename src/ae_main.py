import numpy as np
import pandas as pd 
import os

from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
from datetime import date
import time

# from random import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, LeakyReLU, Lambda
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from keras import backend as K
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.callbacks import TensorBoard
from keras.layers import Activation

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from PIL import Image
import matplotlib


######################
import model

#On charge les poids de l encodeur / decodeur

INPUT_DIM = (64,64,3) # Image dimension
Z_DIM = 512 # Dimension of the latent vector (z)
WEIGHTS_FOLDER = './ae_weights/'
MEAN_VECTOR_FOLDER = './mean_vector_folder/'
input_path = "./input/"

encoder_input, encoder_output,  shape_before_flattening, encoder  = model.build_encoder(input_dim = INPUT_DIM,
                                    output_dim = Z_DIM, 
                                    conv_filters = [32, 64, 64, 64],
                                    conv_kernel_size = [3,3,3,3],
                                    conv_strides = [2,2,2,2])

decoder_input, decoder_output, decoder = model.build_decoder(input_dim = Z_DIM,
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [2,2,2,2]
                                        )
                                        
# The input to the model will be the image fed to the encoder.
simple_autoencoder_input = encoder_input

# The output will be the output of the decoder. The term - decoder(encoder_output) 
# combines the model by passing the encoder output to the input of the decoder.
simple_autoencoder_output = decoder(encoder_output)

# Input to the combined model will be the input to the encoder.
# Output of the combined model will be the output of the decoder.
simple_autoencoder = Model(simple_autoencoder_input, simple_autoencoder_output)

encoder.load_weights(os.path.join(WEIGHTS_FOLDER,"encoder_weights.h5"))
decoder.load_weights(os.path.join(WEIGHTS_FOLDER,"decoder_weights.h5"))
simple_autoencoder.load_weights(os.path.join(WEIGHTS_FOLDER,"autoencoder_weights.hdf5"))


#####################################


#On charge une image

def preprocessing_img(img):  
    return(img.astype(float)/255.0)
    #return(2*(img.astype(float) - 127.5)/255.0)

def _imread(image_name):
#     return plt.imread(image_name)
    return cv2.imread(image_name)

def _imresize(image_array, size):
    return cv2.resize(image_array, dsize=size)



input_list = os.listdir(input_path)
vector_list = os.listdir(MEAN_VECTOR_FOLDER)


for file in input_list:
    img = np.array(Image.open(input_path + file))
    #reshape en (1,64,64,3)
    img = _imresize(img, INPUT_DIM[:2])
    img = preprocessing_img(img)
    img=np.expand_dims( img, axis=0)
    img = img[:, :,  :, :3]
    img_encoded = encoder.predict(img)
    
    for vector in vector_list:
        mean_vector = np.load(MEAN_VECTOR_FOLDER + vector)
        img_decoded = decoder.predict(np.reshape(mean_vector, (1,Z_DIM))+(img_encoded))
        matplotlib.image.imsave('./output/{}.{}.jpg'.format(file,vector), img_decoded[0])    
    os.system("python deblur_gan/deblur_image.py --weight_path=deblur_gan/generator.h5 --input_dir=output --output_dir=output")

