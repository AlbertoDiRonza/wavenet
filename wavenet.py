from keras.layers import (
    Add, Activation, Conv1D, Dense, Flatten, Multiply)
from keras.models import Model

def build_residual_block(num_filters, kernel_size,
                         dilation_rate, x_input):
    """
    Define a residual block for the WaveNet model as described in the paper see: 
    https://arxiv.org/pdf/1609.03499v2
    It consists of gated activation units, residual connection and skip connection.
    
    Args:
        num_filters (int): Number of filters for the convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate for the convolutional layers.
        
    Returns:
    """
    
    # Gated activation
    x_sigmoid = Conv1D(
        num_filters, kernel_size, dilation_rate, 
        padding='same', strides=1, activation='sigmoid')(x_input)
    x_tanh = Conv1D(
        num_filters, kernel_size, dilation_rate, 
        padding='same', strides=1, activation='tanh')(x_input)
    x_mul = Multiply()([x_sigmoid, x_tanh])
    
    # caluculating skip connection and residual connection
    skip_conn = Conv1D(1,1)(x_mul)
    residual_out = Add()([skip_conn, x_mul])
    
    return residual_out, skip_conn


def build_wavenet_model(input_size, num_residual_blocks, 
                num_filters, kernel_size): 
    # Define the input
    x_input = Input(batch_shape=(None, input_size, 1))
    # First convolution over the input
    x_conv1 = Conv1D(num_filters, kernel_size, padding='same')(x_input)
    output_skip_conn = []
    
    # Fulling the skip connection list
    for i in range(num_residual_blocks):
        x_conv1, output_skip_conn = build_residual_block(num_filters, kernel_size, 
                                                         2**(i+1), x_conv1)
        output_skip_conn.append(output_skip_conn)
    
    # Preparing the output of the model
    output = Add()(output_skip_conn)
    output_relu1 = Activation('relu')(output)
    output_conv1 = Conv1D(1, 1)(output_relu1)
    output_relu2 = Activation('relu')(output_conv1)
    output_conv2 = Conv1D(1, 1)(output_relu2)
    
    # The output here is a feature map that has to be flattened and passed to a dense layer
    output_flatten = Flatten()(output_conv2)
    output_softmax = Dense(256, activation='softmax')(output_flatten)
    
    # Declaring a model object from Model class and inizializing it with the input and output layers
    model = Model(inputs=x_input, outputs=output_softmax)
    # https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model