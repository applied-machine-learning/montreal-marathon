from data import *
import numpy as np

def run():
    dataset = MarathonDataset(loadCSV("raw_data/Project1_data.csv"))

    ''' Headers Used:
    - Gender
    - Age
    - log (number of non 2015 marathons / average non 2015 marathon time (handle div by zero))
    '''
    inputs = np.matrix(dataset.request([Headers.gender, Headers.age, Headers.logNon2015MarathonRatio]))
    outputs = np.matrix(dateset.request([Headers.MM2015Time]))



if __name__ == "__main__":
    run()
