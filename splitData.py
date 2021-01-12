import os
import random
from shutil import copyfile


#fungsi mencopy dan men split dataset ke data/training dan data/validation
def split_data(source, training, validation):
    files = []
    for filename in os.listdir(source):
        file = source + "/" + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * .90)
    valid_length = int(len(files) * .10)

    shuffled_set = random.sample(files, len(files))

    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = source + "/" + filename
        destination = training + "/" + filename
        copyfile(this_file, destination)

    for filename in valid_set:
        this_file = source + "/" + filename
        destination = validation + "/" + filename
        copyfile(this_file, destination)