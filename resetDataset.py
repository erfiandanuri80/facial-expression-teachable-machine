import shutil
from pathlib import Path
import os


def resetDataset():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("dataset").mkdir(parents=True, exist_ok=True)
    Path("predict").mkdir(parents=True, exist_ok=True)

    modelFile = "model.h5"

    ## If file exists, delete it ##
    if os.path.isfile(modelFile):
        os.remove(modelFile)
    # removing directory
    shutil.rmtree('data')
    shutil.rmtree('predict')
    shutil.rmtree('dataset')
    print("Data telah direset")