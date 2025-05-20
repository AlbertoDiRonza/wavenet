import os

import numpy as np
import scipy.io.wavfile as wav

# Reading the audio files
def read_audio_file(file_path):
    """
    Reads a ".wav" audio file and returns the audio data [TODO : .read() returns data-type from the file please 
    ta check the type of the data in the dataset
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html].
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        tuple: Sample rate and audio data.
    """
    _, audio_data = wav.read(file_path)
    return audio_data

# The following 'scale_*' functions implement the known formula:
# f(x) = (b - a) * ((x - min) / (max - min)) + a
# For scaling a given value x between the range [a, b].

def scale_audio_uint8_to_float64(arr):
    """
    Scales an array of unsigned 8-bit unsigned integers [0, 255] 
    to a float64 array in the range [-1, 1].
    """
    vmax = np.iinfo(np.uint8).max
    vmin = np.iinfo(np.uint8).min
    arr = arr.astype(np.float64)
    return (arr - vmin) / (vmax - vmin) * 2 - 1

def scale_audio_int16_to_float64(arr):
    """
    Scales an array of signed 16-bit integers [-32768, 32767] 
    to a float64 array in the range [-1, 1].
    """
    vmax = np.iinfo(np.int16).max
    vmin = np.iinfo(np.int16).min
    arr = arr.astype(np.float64)
    return (arr - vmin) / (vmax - vmin) * 2 - 1

def scale_audio_float64_to_uint8(arr):
    """
    Scales an array of float64 values in the range [-1, 1] 
    to an unsigned 8-bit integer array [0, 255].
    Inverse of scale_audio_uint8_to_float64.
    """
    vmax = np.iinfo(np.uint8).max
    arr = ((arr + 1) / 2) * vmax
    arr = arr.astype(np.uint8)
    return arr

def scale_audio_float64_to_int16(arr):
    """
    Scales an array of float64 values in the range [-1, 1] 
    to a signed 16-bit integer array [-32768, 32767].
    Inverse of scale_audio_int16_to_float64.
    """
    vmax = np.iinfo(np.int16).max
    vmin = np.iinfo(np.int16).min
    arr = ((arr + 1) / 2) * (vmax - vmin) + vmin
    arr = arr.astype(np.int16)
    return arr
    
def mu_law_encode(xt, mu=255):
    """
    Applies mu-law encoding to the audio data so that the data is in the range [-1, 1] 
    and it can be quantized in range [0, 255] for softmax output.
    
    Args:
        xt: input audio data to be encoded.
        mu (int): Mu-law parameter.
        
    Returns:
        numpy.ndarray: Mu-law encoded audio data.
    """
    return np.sign(xt) * np.log1p(mu * np.abs(xt)) / np.log1p(mu)

def mu_law_decode(yt, mu=255):
    """
    Applies mu-law decoding to the audio data so that the input is expanded to the original range.
    
    Args:
        xt: input audio data to be decoded.
        mu (int): Mu-law parameter.
        
    Returns:
        numpy.ndarray: Mu-law decoded audio data.
    """
    return np.sign(yt) * (1 / mu) * (((1 + mu) ** np.abs(yt)) - 1)

def to_one_hot(xt):
    """
    Converts the input data to one-hot encoding.
    
    Args:
        xt: input data to be converted.
        
    Returns:
        numpy.ndarray: One-hot encoded data.
    """
    return np.eye(256)[xt.astype(int)]  # Convert to int for indexing

def get_audio_sample_batches(file_path, receptive_field_size, 
                              stride_step=32): 
    """
    Provides the audio samples in batches for training and validation. 
    
    Args:
        file_path (str): Path to the audio file.
        receptive_field_size (int): Size of the receptive field.
        stride_step (int): Step size for the sliding window.
    """
    audio_files = [
        file_path + fn for fn in os.listdir(file_path) 
        if fn.endswith('.wav')]
    
# TODO : capire bene il ruolo di X e y 
    X = []
    y = []
    for audio_file in audio_files:
        # Read the audio file
        audio = read_audio_file(audio_file)
        # Scaliing to from 16 bit to (-1 , 1)
        audio = scale_audio_int16_to_float64(audio)
        # Sliding window
        offset = 0
        while offset + receptive_field_size - 1 < len(audio):
            X.append(
                audio[
                    offset:offset + receptive_field_size
                ].reshape(receptive_field_size, 1)
            )
            # TODO: y è il vettore in input? 
            y_cur = audio[receptive_field_size]
            y_cur = mu_law_encode(y_cur)
            # Scaliing to from float 64 to unsigned int 8 bit
            y_cur = scale_audio_float64_to_uint8(y_cur)
            y.append(to_one_hot(y_cur))
            # Sliding the window
            offset += stride_step
    return np.array(X), np.array(y)
    
def prediction_to_waveform_value(probability_distribution, random=False): 
    """
    Accepts the output of the model as input (a probability vector
    of size 256) and returns a 16 bits integer that corresponds to the
    position selected in the expanded space. 
    
    Args:
        probability_distribution (numpy.ndarray): Probability 
        distribution from the model, it is a 1-d vector.
        random (bool): If True, select a random value between 0 
        and 255 and use it to reconstruct the signal, drawn according
        to the provided distribution. Otherwise, the most probable 
        value will be selected.
    """
    if random: 
        choice = nprandom.choice(
            range(256), 
            p=probability_distribution
        )
    else:
        choice = np.argmax(probability_distribution)
    #scaling the value back to int 16 bit
    y_cur = scale_audio_uint8_to_float64(choice)
    y_cur = mu_law_decode(y_cur)
    y_cur = scale_audio_float64_to_int16(y_cur)
    return y_cur