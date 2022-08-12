import pandas as pd

from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import logging
import os

gate = "AND Gate"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs","running_logs.log"),
    level=logging.INFO,
    format = '[%(asctime)s: %(levelname)s : %(module)s]: %(message)s',
    filemode='a'
)

def main(data, modelName, plotName, eta, epochs):
    # providing inputs for class Perceptron
    df = pd.DataFrame(data)
    logging.info("This is the raw dataset: \n{df}")

    X,y = prepare_data(df)

    # Initializing class Perceptron
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss

    model.save(filename=modelName, model_dir="model")

    # saving the plot
    save_plot(df, model, filename=plotName)

if __name__ == "__main__":
    # data for OR gate
    AND = {'x1':[0,0,1,1], 
        'x2' : [0,1,0,1], 
        'y':[0,0,0,1]}
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info('>>>>> starting training for {gate} >>>>>')
        main(data=AND, modelName='and.model', plotName='and.png',eta=ETA, epochs=EPOCHS)
        logging.info('<<<<< Completed training for {gate} <<<<<\n\n')

    except Exception as e:
        logging.exception(e)
        raise e
    