import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2


def realtimePredict():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation', class_mode='categorical')

    cap = cv2.VideoCapture(0)
    model = load_model('model.h5')
    while (cap.isOpened()):

        ret, frame = cap.read()

        resize = cv2.resize(frame, (150, 150))
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        #faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        X = image.img_to_array(resize)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])

        result = model.predict(images)
        labels = validation_generator.class_indices.items()
        category = []
        for i in labels:
            category.append(i)
        kelas = np.argmax(result)

        for key, value in labels:
            if value == kelas:
                print("kelas", value, key)
                #menambahkan text pada frame
                cv2.putText(frame, key, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

        # Show the image
        cv2.imshow("Identified Face", frame)
        # Wait for user keypress
        key = cv2.waitKey(1) & 0xFF

        if key % 256 == 32:
            break

    cap.release()
    cv2.destroyAllWindows()


def inputPredict(img):

    dim = (150, 150)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation', class_mode='categorical')

    test_image = image.load_img(img, target_size=dim)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    model = load_model('model.h5')
    print("Loaded model from disk")
    result = model.predict(test_image)
    labels = (validation_generator.class_indices)
    category = []
    for i in labels:
        category.append(i)
    kelas = np.argmax(result)
    global predictions
    predictions = category[kelas]
    print(category)
    print(result)
    print(predictions)