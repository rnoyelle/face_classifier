import numpy as np
from PIL import Image
import click
import os

from deblur_gan.deblurgan.model import generator_model
from deblur_gan.deblurgan.utils import load_image, deprocess_image, preprocess_image
        

def deblur(weight_path, input_dir, output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    for image_name in os.listdir(input_dir):
        if image_name != "unblurred":
            image = np.array([preprocess_image(load_image(input_dir+'/'+image_name))])
            x_test = image
            x_test= x_test[:, :, :,:3]
            generated_images = g.predict(x=x_test)
            generated = np.array([deprocess_image(img) for img in generated_images])
            x_test = deprocess_image(x_test)
            for i in range(generated_images.shape[0]):
                x = x_test[i, :, :, :]
                img = generated[i, :, :, :]
                #output = np.concatenate((x, img), axis=1)        #Si on veut concatèner l'image avant/après défloutage
                #im = Image.fromarray(output.astype(np.uint8))
                im = Image.fromarray(img.astype(np.uint8)) 
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                im.save(os.path.join(output_dir, image_name))

weight_path = "deblur_gan/generator.h5"
folder_list = os.listdir('output/modified/autoencoder/')
for folder in folder_list :
    if folder != "unblurred":
        print("unblurring : "+folder+ '...')
        deblur(weight_path, 'output/modified/autoencoder/'+folder, 'output/modified/autoencoder/'+folder+'/unblurred/')

