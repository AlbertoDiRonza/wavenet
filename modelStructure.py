from keras.utils import plot_model
from keras.models import load_model

def plot_model_structure(path_to_model):
    wavenet = load_model(path_to_model)
    plot_model(
        wavenet,
        to_file='model_structure.png',
        show_shapes=True,
    )
    print("Model structure saved as 'model_structure.png'")
    return