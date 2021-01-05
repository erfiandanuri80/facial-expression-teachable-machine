import index
import matplotlib.pyplot as plt


#FUNGSI TRAINING MODEL
def trainingModel(epochs, batch_size, num_class):
    #IMPORT LIBRARY
    import tensorflow as tf
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras import backend as K
    import os

    img_width, img_height = 150, 150

    train_data_dir = 'data/training'
    valid_data_dir = 'data/validation'
    #test_data_dir = 'data/test'

    ## ARSITEKTUR
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    #input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

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

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_datagen = ImageDataGenerator(rescale=1 / 255.0)
    validation_generator = validation_datagen.flow_from_directory(
        valid_data_dir,
        batch_size=batch_size,
        class_mode='categorical',
        target_size=(img_height, img_width))

    history = model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
    )

    model_save_weight = model.save_weights('weight.h5')
    model_save = model.save('model.h5')

    global acc, val_acc, loss, val_loss

    acc = history.history["accuracy"]
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(acc, val_acc, loss, val_loss)


def graphAccuracy(acc, val_acc, epochs):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(epochs, acc, 'r', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('Accuracy_curve_CNN.jpg')


#trainingModel(1,16,5)
