from wavenet.wavenet import build_wavenet_model
from wavenet.preprocess import get_audio_sample_batches

def train_wavenet(path_to_audio_train, path_to_audio_val, input_size,
                  num_filters, kernel_size, num_residual_blocks,
                  batch_size, num_epochs): 
    """
    Trains a WaveNet model on the given audio files for training and validation.
    
    Args:
        path_to_audio_train (str): Path to the training audio files.
        path_to_audio_val (str): Path to the validation audio files.
        input_size (int): Size of the input data.
        num_filters (int): Number of filters for convolution in the causal and dilated 
            convolution layers.
        kernel_size (int): Size of the convolutional window for causal in the dilated c
            convolution layers.
        num_residual_blocks (int): Number of residual blocks to generate between input and output.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
    
   
    """
    wavenet = build_wavenet_model(
        input_size, num_filters, kernel_size, num_residual_blocks)
    print("Generating training data...")
    X_train, y_train = get_audio_sample_batches(path_to_audio_train, input_size)
    print("Generating validation data...")
    X_val, y_val = get_audio_sample_batches(path_to_audio_val, input_size)
    print("Training the model...")
    history = wavenet.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=num_epochs
    )
    return wavenet, history

    distinct_filename = "wavenet_in{}_nf{}_k{}_nres{}_bat{}_e{}.h5".format(
        input_size, num_filters, 
        kernel_size, 
        num_residual_blocks,
        batch_size, 
        num_epochs)
    
def main():
    path_to_audio_train = 
    path_to_audio_val =
    input_size = 1024
    num_filters = 16
    kernel_size = 2
    num_residual_blocks = 9
    batch_size = 32
    num_epochs = 20
    
    wavenet, history = train_wavenet(
        path_to_audio_train, path_to_audio_val, input_size,
        num_filters, kernel_size, num_residual_blocks,
        batch_size, num_epochs)
    
    # Save the trained model
    
    
if __name__ == "__main__":
    main()