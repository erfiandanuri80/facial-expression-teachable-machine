import index
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#FUNGSI TRAINING MODEL
def trainingModel(epochs, batch_size, num_class):
    #IMPORT LIBRARY
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense

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
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(num_class, activation='softmax'))
    #model.summary()

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #TRANSFORMASI DATA (IMAGE AUGMENTATION)
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

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

    #model_save_weight = model.save_weights('weight.h5')
    model_save = model.save('model.h5')
    print("Saved model to disk")
