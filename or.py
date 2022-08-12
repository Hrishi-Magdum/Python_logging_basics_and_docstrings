import pandas as pd

from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron

def main(data, modelName, plotName, eta, epochs):
    # providing inputs for class Perceptron
    df_OR = pd.DataFrame(data)

    X,y = prepare_data(df_OR)

    # Initializing class Perceptron
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss

    model.save(filename=modelName, model_dir="model")

    # saving the plot
    save_plot(df_OR, model, filename=plotName)

if __name__ == "__main__":
    # data for OR gate
    OR = {'x1':[0,0,1,1], 
        'x2' : [0,1,0,1], 
        'y':[0,1,1,1]}
    ETA = 0.3
    EPOCHS = 10
    main(data=OR, modelName='or.model', plotName='or.png',eta=ETA, epochs=EPOCHS)