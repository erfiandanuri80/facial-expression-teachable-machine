import matplotlib.pyplot as plt
import os


#FUNGSI MENAMPILKAN GRAPH DISTRIBUSI DATA TRAINING DG MATPLOTLIB.PYPLOT
def showDataTraining(source):
    dir = []
    for subdirs in os.walk(source):
        dir.append(subdirs)

    if dir == []:
        message = 'insert data please!'
        print(message)
    else:
        image_class = dir[0][1]
        num_images = {}
        for i in image_class:
            len_images = len(os.listdir('data/training/' + i + '/'))
            num_images[i] = len_images
        plt.figure(figsize=(6, 5))
        plt.bar(range(len(num_images)),
                list(num_images.values()),
                align='center')
        plt.xticks(range(len(num_images)), list(num_images.keys()))
        plt.title('Distribution of different classes in Training Dataset')
        plt.show()


#FUNGSI MENAMPILKAN GRAPH DISTRIBUSI DATA VALIDATION DG MATPLOTLIB.PYPLOT
def showDataValidation(source):
    dir = []
    for subdirs in os.walk(source):
        dir.append(subdirs)

    if dir == []:
        message = 'insert data please!'
        print(message)
    else:
        image_class = dir[0][1]

        num_images = {}
        for i in image_class:
            len_images = len(os.listdir('data/validation/' + i + '/'))
            num_images[i] = len_images
        plt.figure(figsize=(6, 5))
        plt.bar(range(len(num_images)),
                list(num_images.values()),
                align='center')
        plt.xticks(range(len(num_images)), list(num_images.keys()))
        plt.title('Distribution of different classes in Validation Dataset')
        plt.show()