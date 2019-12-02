"""
File: train_age_classifier.py
Author: Rudy
Description: Train age classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from utils.datasets import DataManager
from models.cnn import classifier_VGG16
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data

# parameters
batch_size = 10
num_epochs = 10
validation_split = .2
do_random_crop = False
patience = 100
num_classes = 101
dataset_name = 'imdb_age'
input_shape = (256, 256, 3)
if input_shape[2] == 1:
    grayscale = True
else :
    grayscale = False
base_path = '/donnees/rnoyelle/deep_annotation/' # 'C:/Users/Rudy/Documents/Projet_Digitale/deep_annotation/'
images_path = base_path + 'datasets/imdb_crop/' # base_path + 'datasets/imdb_crop/' # 'C:/Users/Rudy/Documents/Projet_Digitale/dataset/IMDB/imdb_crop/' #base_path + 'datasets/imdb_crop/'
log_file_path = base_path + 'models/dev_models/age_models/gender_training.log'
trained_models_path = base_path + 'models/dev_models/age_models/age_classifier_VGG16'

print('bouh')


# data loader
data_loader = DataManager(dataset_name, dataset_path=images_path + 'imdb.mat')
ground_truth_data = data_loader.get_data()
# subset_keys = sorted(ground_truth_data.keys())[:60]
ground_truth_data =  {key : ground_truth_data[key] for key in subset_keys}
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))
image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 norm_input='vgg',
                                 num_classes=num_classes,
                                 do_random_crop=do_random_crop)


# model parameters/compilation

model = classifier_VGG16(input_shape, num_classes)
model.get_layer('conv_layer').trainable = False
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                            patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))
