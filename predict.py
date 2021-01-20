import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2


def realtimePredict():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation', class_mode='categorical')

    cap = cv2.VideoCapture(0)
    model = load_model('model.h5')
    labels = validation_generator.class_indices
    category = []
    for i in labels:
        category.append(i)
    while (cap.isOpened()):

        ret, frame = cap.read()

        resize = cv2.resize(frame, (150, 150))
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face

        X = image.img_to_array(resize)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])

        for (x, y, w, h) in faces:
            result = model.predict(images)
            kelas = np.argmax(result)
            predictions = category[kelas]
            print(predictions, kelas, result)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (1, 1, 1), 1)
            # menambahkan text pada frame
            cv2.putText(frame, predictions, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            x_offset = x
            y_offset = y - (h // 3)
            if y_offset < 0:
                y_offset = y

            s_img = cv2.imread("emoticon/{}.png".format(predictions))

            s_img = cv2.resize(s_img, (w, h // 4))

            y1, y2 = y_offset, y_offset + s_img.shape[0]
            x1, x2 = x_offset, x_offset + s_img.shape[1]

            # alpha_s = s_img[:, :, 2] / 255
            # alpha_l = 1.0 - alpha_s

            frame[y_offset:y_offset + s_img.shape[0],
                  x_offset:x_offset + s_img.shape[1]] = s_img
            #for c in range(0, 3):
            #    frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
            #                              alpha_l * frame[y1:y2, x1:x2, c])

        #for key, value in labels:
        #    if value == kelas:
        #        print("kelas", value, key)

        # Show the image
        cv2.imshow("Identified Face", frame)
        # Wait for user keypress
        if cv2.waitKey(1) != -1:
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