import index
import matplotlib.pyplot as plt


def resetDataset():
    import shutil
    from pathlib import Path
    import os

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("dataset").mkdir(parents=True, exist_ok=True)
    Path("predict").mkdir(parents=True, exist_ok=True)
    directory = 'data'
    directory2 = 'predict'
    directory3 = 'dataset'
    # removing directory
    shutil.rmtree(directory)
    shutil.rmtree(directory2)
    shutil.rmtree(directory3)


#FUNGSI TRAINING MODEL
def trainingModel(epochs, batch_size, num_class):
    #IMPORT LIBRARY

    from keras.models import model_from_json
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras import backend as K
    import tensorflow as tf
    import os

    #PARAMETER INPUT UNTUK NETWORK
    dim = (150, 150)
    channel = (3, )
    input_shape = dim + channel

    #batch_size
    batch_size = batch_size
    epochs = epochs
    num_class = num_class

    #DIREKTORI
    train_data_dir = 'data/training'
    valid_data_dir = 'data/validation'

    ## ARSITEKTUR

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    #model.summary()

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #TRANSFORMASI DATA (IMAGE AUGMENTATION)
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)

    #FLOW DATA
    # categorical = 1,2,3,4,5

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=dim,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    validation_generator = validation_datagen.flow_from_directory(
        valid_data_dir,
        batch_size=batch_size,
        target_size=dim,
        class_mode='categorical',
        shuffle=True)

    num_class = validation_generator.num_classes
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        shuffle=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_save = model.save_weights("model.h5")
    print("Saved model to disk")

    #model_save_weight = model.save_weights('weight.h5')
    #model_save = model.save('model.h5')

    global acc, val_acc, loss, val_loss
    acc = history.history["accuracy"]
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("accuracy", acc, val_acc, "loss", loss, val_loss)

    return model


def predict(img):
    import numpy as np
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import model_from_json

    dim = (150, 150)
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    dim = (150, 150)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        batch_size=16,
        target_size=dim,
        class_mode='categorical',
        shuffle=True)

    test_image = image.load_img(img, target_size=dim)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = loaded_model.predict(test_image)
    labels = (validation_generator.class_indices)
    category = []
    for i in labels:
        category.append(i)
    kelas = np.argmax(result)
    global predictions
    predictions = category[kelas]
    print(predictions)