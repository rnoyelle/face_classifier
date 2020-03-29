import numpy as np
import pandas as pd
import os

from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt

from datetime import date
from datetime import datetime
from tqdm import tqdm
import csv

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras import Input, Model
# from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization
from keras.layers import Reshape, UpSampling2D, concatenate, LeakyReLU, Lambda
from keras.layers import Activation
from keras.optimizers import Adam
from keras import backend as K
# from keras.layers import MaxPooling2D
# from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
# from keras.callbacks import TensorBoard
# import time
# from random import shuffle

"""
paths
"""
# base_path = '/donnees/rnoyelle/deep_annotation/' # 'C:/Users/Rudy/Documents/Projet_Digitale/deep_annotation/'
# images_path = 'C:/Users/Rudy/Documents/Projet_Digitale/dataset/IMDB/imdb_crop/'
images_path = '/donnees/rnoyelle/deep_annotation/datasets/imdb_crop/'

"""
load raw data
"""
#
face_score_treshold = 3

# load mat file
dataset = loadmat(images_path + 'imdb.mat')

# extract info
image_names_array = dataset['imdb']['full_path'][0, 0][0]
dob_array = dataset['imdb']['dob'][0, 0][0]  # dob = date of birth
photo_date_array = dataset['imdb']['photo_taken'][0, 0][0]
gender_classes = dataset['imdb']['gender'][0, 0][0]
celeb_id_array = dataset['imdb']['celeb_id'][0, 0][0]

# create mask
face_score = dataset['imdb']['face_score'][0, 0][0]
second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
face_score_mask = face_score > face_score_treshold
second_face_score_mask = np.isnan(second_face_score)
unknown_dob_mask = np.logical_not(np.logical_or(np.isnan(dob_array), np.isnan(photo_date_array)))
mask = np.logical_and(face_score_mask, second_face_score_mask)
mask = np.logical_and(mask, unknown_dob_mask)

# apply mask
image_names_array = image_names_array[mask]
dob_array = dob_array[mask]
photo_date_array = photo_date_array[mask]
gender_classes = gender_classes[mask]
celeb_id_array = celeb_id_array[mask]

print(image_names_array.shape, image_names_array[0])
print(dob_array.shape, dob_array[0])
print(photo_date_array.shape, photo_date_array[0])
print(gender_classes.shape, gender_classes[0])
print(celeb_id_array.shape, celeb_id_array[0])

"""
prepossessing data
"""
df = pd.DataFrame(data={'image_names': [el[0] for el in image_names_array],
                        'dob': dob_array,
                        'photo_date': photo_date_array,
                        'gender_classes': gender_classes,
                        'celeb_id': celeb_id_array})

df.dropna(axis=0, inplace=True)


def calculate_age_in_year(dob, photo_date):
    dob = date.fromordinal(dob)
    photo_date = date(year=photo_date,
                      month=7,
                      day=1)
    age = int((photo_date - dob).days // 365)
    return age

df['age'] = df.apply(lambda row: calculate_age_in_year(row['dob'], row['photo_date']), axis=1)
df = df[(df['age'] <= 100) & (df['age'] > 0)]
# bins_age = [0, 20, 30, 40, 50, 60, 100]
bins_age = [0, 20, 40, 60, 100]
df.loc[:, 'age_cat'] = pd.cut(df['age'], bins_age, labels=False, right=False)
# pd.cut(df['age'], [0, 20, 25, 32, 38, 45, 60], labels=False, right=False)

# split into train, val, test : 0.7, 0.2, 0.1
random_state = 42
celeb_id_train, celeb_id_val = train_test_split(df['celeb_id'].unique(), test_size=0.2,   random_state=random_state)
celeb_id_train, celeb_id_test = train_test_split(celeb_id_train,         test_size=0.125, random_state=random_state)
df.loc[df['celeb_id'].isin(celeb_id_train), 'set'] = 'train'
df.loc[df['celeb_id'].isin(celeb_id_val),   'set'] = 'val'
df.loc[df['celeb_id'].isin(celeb_id_test),  'set'] = 'test'

df.drop(['dob', 'photo_date', 'celeb_id', 'age'], axis=1, inplace=True)
num_cat_age = df['age_cat'].nunique()
df = df[df['set'] == 'train']
# df = df.head(20000)
print(num_cat_age)

"""
Network params (inputs/outputs)
"""
image_shape = (64, 64, 3)  # input image size
input_shape = image_shape
z_dim = 100  # latents dim
# n_y = num_cat_age + 2  # labels dim
n_y = num_cat_age  # labels dim

"""
Data Loader
"""


def preprocessing_img(img):
    return 2.0*(img.astype(float) - 127.5)/255.0


def _imread(image_name):
    # return plt.imread(image_name)
    return cv2.imread(image_name)


def _imresize(image_array, size):
    return cv2.resize(image_array, dsize=size)


def image_generator(mode='train'):
    """
    mode : train, val, test
    """
    X = []
    y_age = []
    y_gender = []
    while True:
        for idx, row in df[df['set'] == mode].iterrows():
            # images
            image_path = os.path.join(images_path, row['image_names'])
            image_array = _imread(image_path)
            image_array = _imresize(image_array, image_shape[:2])
            image_array = preprocessing_img(image_array)
            X.append(image_array)

            # labels
            y_gender.append(row['gender_classes'])
            y_age.append(row['age_cat'])

            if len(X) == batch_size:
                # images
                X = np.asarray(X)

                # labels
                y_age = to_categorical(y_age, num_classes=num_cat_age)
                # y_gender = np.array(y_gender).reshape(-1, 1)
                # y_gender = to_categorical(y_gender, num_classes=2)
                # Y = np.hstack((y_age, y_gender))

                yield X, y_age
                X = []
                y_age = []
                y_gender = []

#         if len(X) > 0:
#             # images
#             X = np.asarray(X)

#             # labels
#             y_age = to_categorical(y_age, num_classes=num_cat_age)
#             y_gender = np.array(y_gender).reshape(-1, 1)
#             y_gender = to_categorical(y_gender, num_classes=2)
#             Y = np.hstack((y_age, y_gender))

#             yield X, Y


"""
Learning params
"""
num_epochs = 100  # 500
# patience = 2
batch_size = 64
# opt = 'adam'
dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

"""
build models
"""


def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 32, 32, 1])
    return x


def build_discriminator():
    image_input = Input(shape=input_shape)
    label_input = Input(shape=(n_y,))

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    label_input1 = Lambda(expand_label_input)(label_input)
    x = concatenate([x, label_input1], axis=3)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model


def build_generator():
    input_z_noise = Input(shape=(z_dim,))
    input_label = Input(shape=(n_y,))

    x = concatenate([input_z_noise, input_label])

    x = Dense(2048, input_dim=z_dim + n_y)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model


def save_rgb_img(img, path):
    """
    Save an RGB image
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(img)
    ax.imshow((img+1.0)/2.0)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


"""
Build and compile the discriminator network
"""
# epoch0 = 2
discriminator = build_discriminator()
# discriminator.load_weights("discriminator_epoch{}.h5".format(epoch0))
discriminator.compile(loss=['binary_crossentropy'],
                      optimizer=dis_optimizer)

"""
Build and compile the generator network
"""
generator = build_generator()
# generator.load_weights("generator_epoch{}.h5".format(epoch0))
generator.compile(loss=['binary_crossentropy'],
                  optimizer=gen_optimizer)

"""
Build and compile the adversarial model
"""
discriminator.trainable = False
input_z_noise = Input(shape=(z_dim, ))
input_label = Input(shape=(n_y, ))
recons_images = generator([input_z_noise, input_label])
valid = discriminator([recons_images, input_label])
adversarial_model = Model(inputs=[input_z_noise, input_label],
                          outputs=[valid])
adversarial_model.compile(loss=['binary_crossentropy'],
                          optimizer=gen_optimizer)

"""
some comment
"""
# create log file
t = datetime.today().strftime('%Y_%m_%d')
filename = 'logs/{}.log'.format(t)
file_exists = os.path.isfile(filename)
with open(filename, 'a') as csvfile:
    headers = ['epoch', 'i_batch', 'loss_D', 'loss_G', 'D_pos', 'D_neg']
    writer = csv.DictWriter(csvfile, delimiter='\t', lineterminator='\n', fieldnames=headers)
    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header

dir_exists = os.path.exists('results/{}'.format(t))
if not dir_exists:
    os.makedirs('results/{}'.format(t))
dir_exists = os.path.exists('models/{}'.format(t))
if not dir_exists:
    os.makedirs('models/{}'.format(t))


# serialize model to JSON
model_json = discriminator.to_json()
with open("models/{}/discriminator.json".format(t), "w") as json_file:
    json_file.write(model_json)

model_json = generator.to_json()
with open("models/{}/generator.json".format(t), "w") as json_file:
    json_file.write(model_json)


with open("models/{}/labels.csv".format(t), 'a') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='\t', lineterminator='\n', fieldnames=['age_bins', 'gender'])
    writer.writeheader()
    writer.writerow({'age_bins': bins_age, 'gender': False})

"""
Training
"""
# Implement label smoothing
real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

for epoch in range(0, num_epochs):
    Dpos = []
    Dneg = []
    gen_losses = []
    dis_losses = []
    n = int(df[df['set'] == 'train'].shape[0] / batch_size)
    img_gen = image_generator(mode='train')
    for i_batch in tqdm(range(n)):
        images_batch, y_batch = next(img_gen)
        # print('y shape', y_batch.shape)
        z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))

        """
        Generate fake images
        """
        initial_recons_images = generator.predict_on_batch([z_noise, y_batch])

        d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
        d_loss_fake = discriminator.train_on_batch([initial_recons_images, y_batch], fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print("d_loss: {}".format(d_loss))

        """
        Train the generator network
        """
        z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
        #         random_labels = np.random.randint(0, n_y, batch_size).reshape(-1, 1)
        #         random_labels = to_categorical(random_labels, n_y)

        random_age = np.random.randint(0, num_cat_age, batch_size).reshape(-1, 1)
        random_age = to_categorical(random_age, num_classes=num_cat_age)
        # random_gender = np.random.randint(0, 2, batch_size).reshape(-1, 1)
        # random_gender = to_categorical(random_gender, num_classes=2)
        random_labels = random_age # np.hstack((random_age, random_gender))

        g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], [1] * batch_size)

        print("g_loss: {}".format(g_loss))

        gen_losses.append(g_loss)
        dis_losses.append(d_loss)

        """
        some comment
        """
        images_batch, y_batch = next(img_gen)
        Dpos.append(np.mean(discriminator.predict_on_batch([images_batch, y_batch])))

        z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
        random_age = np.random.randint(0, num_cat_age, batch_size).reshape(-1, 1)
        random_age = to_categorical(random_age, num_classes=num_cat_age)
        # random_gender = np.random.randint(0, 2, batch_size).reshape(-1, 1)
        # random_gender = to_categorical(random_gender, num_classes=2)
        random_labels = random_age  # np.hstack((random_age, random_gender))

        Dneg.append(np.mean(adversarial_model.predict_on_batch([z_noise2, random_labels])))

        if i_batch % 100 == 0:
            with open(filename, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, delimiter='\t', lineterminator='\n', fieldnames=headers)
                writer.writerow({'epoch': epoch, 'i_batch': i_batch,
                                 'loss_D': np.mean(dis_losses), 'loss_G': np.mean(gen_losses),
                                 'D_pos': np.mean(Dpos), 'D_neg': np.mean(Dneg)})

    """
    Generate and save images
    """
    # images_batch, y_batch
    z_noise = np.random.normal(0, 1, size=(num_cat_age, z_dim))
    y_batch = np.arange(num_cat_age)
    y_batch = to_categorical(y_batch, num_classes=num_cat_age)
    gen_images = generator.predict_on_batch([z_noise, y_batch])

    for i, img in enumerate(gen_images):
        save_rgb_img(img, path="results/{}/img_{}_{}.png".format(t, epoch, i))

    """
    Save networks
    """
    try:
        generator.save_weights("models/{}/{}_generator_epoch{}.h5".format(t, t, epoch))
        discriminator.save_weights("models/{}/{}_discriminator_epoch{}.h5".format(t, t, epoch))
    except Exception as e:
        print("Error: ", e)
