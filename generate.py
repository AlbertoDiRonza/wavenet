import numpy as np 
import scipy.io.wavfile as wav
from keras.models import load_model

from wavenet.preprocess import prediction_to_waveform_value
from modelStructure import plot_model_structure

def generate_audio_audio_model_output(
    path_to_model, path_to_output_wav, input_audio_size,
    genereted_frames, sample_rate):
    
    """
    Generates audio from the model output.
    
    Args:
        path_to_model (str): Path to the trained model.
        path_to_output_wav (str): Path to save the generated audio.
        input_audio_size (int): Size of the input data.
        genereted_frames (int): Number of frames to generate.
        sample_rate (int): Sample rate for the generated audio.
    """
    # Load the trained model
    wavenet = load_model(path_to_model)
    # Initialize the input data
    generated_audio = np.zeros(input_audio_size, dtype=np.int16)
    cur_frame = 0 
    while cur_frame < genereted_frames:
        print(f"cur_frame: {cur_frame} on {genereted_frames}")
        # Remember to check: https://keras.io/api/models/model_training_apis/
        probability_distribution = wavenet.predict(
            generated_audio[cur_frame:].reshape(
                1, input_audio_size, 1)).flatten()
        cur_sample = prediction_to_waveform_value(probability_distribution)
        generated_audio = np.append(
            generated_audio, cur_sample)
        cur_frame += 1
    return generated_audio

def main(): 
    path_to_model = "wavenet_in1024_nf16_k2_nres9_bat32_e20.h5"
    path_to_output_wav = "wavenet_sample_output.wav"
    generated_frames = 2 ** 17 
    input_audio_size = 1024
    
    # Sample rate has to be the same as the files used fot training
    sample_rate = 22050
    
    # Plot the model structure
    plot_model_structure(path_to_model)
    
    generated_audio = generate_audio_audio_model_output(
        path_to_model, path_to_output_wav, input_audio_size,
        generated_frames, sample_rate)
    wav.write(path_to_output_wav, sample_rate, generated_audio)
    
if __name__ == "__main__":
    main()