# We'll need this to check if Tensorflow has access to the GPU
# Useful for debugging during GPU training
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))