from data import *
import numpy as np

def run():
    dataset = MarathonDataset(loadCSV("raw_data/Project1_data.csv"))

    ''' Headers Used:
    - Gender
    - Age
    - log (number of non 2015 marathons / average non 2015 marathon time (handle div by zero))
    '''
    headers = [Headers.gender, Headers.age, Headers.averageNon2015FullMMTime]

    inputs = np.matrix(dataset.request(headers))
    inputs = np.insert(inputs, 3, 1, axis=1)

    outputs = np.matrix(dataset.request([Headers.MM2015Time]))

    weights = np.linalg.inv(np.transpose(inputs) * inputs) * (np.transpose(inputs) * outputs)

    predictions = inputs * weights

    err = outputs - predictions

    sse = np.transpose(err) * err
    
    print weights
    print predictions
    print outputs
    print sse

if __name__ == "__main__":
    run()
