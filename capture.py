import cv2
from pathlib import Path
import splitData


#fungsi mengambil dataframe image ke class
def capturingFrame(formatName):
    cap = cv2.VideoCapture(0)
    count = 0
    Path("dataset").mkdir(parents=True, exist_ok=True)
    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        Path("dataset/{}".format(formatName)).mkdir(parents=True,
                                                    exist_ok=True)
        cv2.imwrite(
            "dataset/{}/{}_{}.jpg".format(formatName, formatName, count),
            frame)
        count = count + 1
        print(count)

        # Display the resulting frame
        cv2.imshow('{}'.format(formatName), frame)

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/training").mkdir(parents=True, exist_ok=True)
    Path("data/validation").mkdir(parents=True, exist_ok=True)

    Path("data/training/{}".format(formatName)).mkdir(parents=True,
                                                      exist_ok=True)
    Path("data/validation/{}".format(formatName)).mkdir(parents=True,
                                                        exist_ok=True)

    #dan setelah mendapat dataset langsung dijalankan fungsi split data nya
    splitData.split_data("dataset/{}".format(formatName),
                         "data/training/{}".format(formatName),
                         "data/validation/{}".format(formatName))


def singleCapture():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('predict', frame)

        Path("predict").mkdir(parents=True, exist_ok=True)
        k = cv2.waitKey(1)

        #tombol spasi
        if k % 256 == 32:
            cv2.imwrite("predict/predict.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()
