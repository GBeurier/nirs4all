import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    DepthwiseConv1D,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    SeparableConv1D,
    SpatialDropout1D,
)

from keras.models import Model, Sequential
from nirs4all.core.utils import framework


@framework('tensorflow')
def decon(input_shape, params={}):
    """
    Builds a CNN model with depthwise and separable convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        64,
        kernel_size=3,
        depth_multiplier=1,
        padding="same",
        activation="relu"
    ))
    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(0.1))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def decon_Sep(input_shape, params={}):
    """
    Builds a CNN model with separable convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(SeparableConv1D(
        filters=params.get('filters1', 64),
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 2),
        depth_multiplier=params.get('depth_multiplier1', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters2', 64),
        kernel_size=params.get('kernel_size2', 3),
        strides=params.get('strides2', 2),
        depth_multiplier=params.get('depth_multiplier2', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters3', 64),
        kernel_size=params.get('kernel_size3', 3),
        depth_multiplier=params.get('depth_multiplier3', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters4', 64),
        kernel_size=params.get('kernel_size4', 3),
        depth_multiplier=params.get('depth_multiplier4', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(Conv1D(
        filters=params.get('filters5', 32),
        kernel_size=params.get('kernel_size5', 5),
        strides=params.get('strides5', 2),
        padding="same",
        activation="relu"
    ))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=params.get('dense_units', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


@framework('tensorflow')
def nicon(input_shape, params={}):
    """
    Builds a custom CNN model with depthwise convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    return model


@framework('tensorflow')
def customizable_nicon(input_shape, params={}):
    """
    Builds a custom CNN model with depthwise convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=params.get('kernel_size1', 15), strides=params.get('strides1', 5), activation=params.get('activation1', "selu")))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=params.get('kernel_size2', 21), strides=params.get('strides2', 3), activation=params.get('activation2', "relu")))
    model.add(BatchNormalization() if params.get('normalization_method1', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=params.get('kernel_size3', 5), strides=params.get('strides3', 3), activation=params.get('activation3', "elu")))
    model.add(BatchNormalization() if params.get('normalization_method2', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation=params.get('dense_activation', "sigmoid")))
    model.add(Dense(1, activation="sigmoid"))
    return model

nicon_sample_finetune = {
    'spatial_dropout': (float, 0.01, 0.5),
    'filters1': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size1': [3, 5, 7, 9, 11, 13, 15],
    'strides1': [1, 2, 3, 4, 5],
    'activation1': ['relu', 'selu', 'elu', 'swish'],
    'dropout_rate': (float, 0.01, 0.5),
    'filters2': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size2': [3, 5, 7, 9, 11, 13, 15],
    'strides2': [1, 2, 3, 4, 5],
    'activation2': ['relu', 'selu', 'elu', 'swish'],
    'normalization_method1': ['BatchNormalization', 'LayerNormalization'],
    'filters3': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size3': [3, 5, 7, 9, 11, 13, 15],
    'strides3': [1, 2, 3, 4, 5],
    'activation3': ['relu', 'selu', 'elu', 'swish'],
    'normalization_method2': ['BatchNormalization', 'LayerNormalization'],
    'dense_units': [4, 8, 16, 32, 64, 128, 256],
    'dense_activation': ['relu', 'selu', 'elu', 'swish'],
}


@framework('tensorflow')
def nicon_VG(input_shape, params={}):
    """
    Builds a custom CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters1', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters3', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters4', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Flatten())
    model.add(Dense(units=params.get('dense_units1', 1024), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units2', 1024), activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


@framework('tensorflow')
def customizable_decon(input_shape, params={}):
    """
    Builds a model using depthwise separable convolutions and layer normalization.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled deconvolutional model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # First block
    model.add(SpatialDropout1D(params.get('spatial_dropout1', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 7), 
        padding=params.get('padding1', "same"), 
        depth_multiplier=params.get('depth_multiplier1', 2), 
        activation=params.get('activationDCNN1', "relu")
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size2', 7), 
        padding=params.get('padding2', "same"), 
        depth_multiplier=params.get('depth_multiplier2', 2), 
        activation=params.get('activationDCNN2', "relu")
    ))
    model.add(MaxPooling1D(pool_size=params.get('pool_size1', 2), strides=params.get('strides1', 2)))
    model.add(LayerNormalization())

    # Second block
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size3', 5), 
        padding=params.get('padding3', "same"), 
        depth_multiplier=params.get('depth_multiplier3', 2), 
        activation=params.get('activationDCNN3', "relu")
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size4', 5), 
        padding=params.get('padding4', "same"), 
        depth_multiplier=params.get('depth_multiplier4', 2), 
        activation=params.get('activationDCNN4', "relu")
    ))
    model.add(MaxPooling1D(pool_size=params.get('pool_size2', 2), strides=params.get('strides2', 2)))
    model.add(LayerNormalization())

    # Third block
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size5', 9), 
        padding=params.get('padding5', "same"), 
        depth_multiplier=params.get('depth_multiplier5', 2), 
        activation=params.get('activationDCNN5', "relu")
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size6', 9), 
        padding=params.get('padding6', "same"), 
        depth_multiplier=params.get('depth_multiplier6', 2), 
        activation=params.get('activationDCNN6', "relu")
    ))
    model.add(MaxPooling1D(pool_size=params.get('pool_size3', 2), strides=params.get('strides3', 2)))
    model.add(LayerNormalization())

    # Final convolution and pooling block
    model.add(SeparableConv1D(
        filters=params.get('separable_filters', 64), 
        kernel_size=params.get('separable_kernel_size', 3), 
        depth_multiplier=params.get('separable_depth_multiplier', 1), 
        padding=params.get('separable_padding', "same"), 
        activation=params.get('activationCNN1', "relu")
    ))
    model.add(Conv1D(
        filters=params.get('conv_filters', 32), 
        kernel_size=params.get('conv_kernel_size', 3), 
        padding=params.get('conv_padding', "same")
    ))
    model.add(MaxPooling1D(pool_size=params.get('final_pool_size', 5), strides=params.get('final_pool_strides', 3)))
    model.add(SpatialDropout1D(params.get('spatial_dropout2', 0.1)))
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(units=params.get('dense_units1', 128), activation=params.get('activationDense1', "relu")))
    model.add(Dense(units=params.get('dense_units2', 32), activation=params.get('activationDense2', "relu")))
    model.add(Dropout(params.get('dropout_rate', 0.2)))

    # Output layer
    model.add(Dense(units=params.get('output_units', 1), activation=params.get('activationDense3', "sigmoid")))

    return model

decon_sample_finetune = {
    'spatial_dropout1': (float, 0.01, 0.5),  # Range for first spatial dropout rate
    'kernel_size1': [3, 5, 7, 9, 11, 13, 15],  # Kernel sizes for the first DepthwiseConv1D
    'padding1': ['same', 'valid'],  # Padding options for the first DepthwiseConv1D
    'depth_multiplier1': [1, 2, 4, 8],  # Depth multiplier for the first DepthwiseConv1D
    'activationDCNN1': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the first DepthwiseConv1D

    'kernel_size2': [3, 5, 7, 9, 11, 13, 15],  # Kernel sizes for the second DepthwiseConv1D
    'padding2': ['same', 'valid'],  # Padding options for the second DepthwiseConv1D
    'depth_multiplier2': [1, 2, 4, 8],  # Depth multiplier for the second DepthwiseConv1D
    'activationDCNN2': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the second DepthwiseConv1D

    'pool_size1': [2, 3, 4, 5],  # Pool sizes for the first MaxPooling1D
    'strides1': [1, 2, 3],  # Stride values for the first MaxPooling1D

    'kernel_size3': [3, 5, 7, 9],  # Kernel sizes for the third DepthwiseConv1D
    'padding3': ['same', 'valid'],  # Padding options for the third DepthwiseConv1D
    'depth_multiplier3': [1, 2, 4, 8],  # Depth multiplier for the third DepthwiseConv1D
    'activationDCNN3': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the third DepthwiseConv1D

    'kernel_size4': [3, 5, 7, 9],  # Kernel sizes for the fourth DepthwiseConv1D
    'padding4': ['same', 'valid'],  # Padding options for the fourth DepthwiseConv1D
    'depth_multiplier4': [1, 2, 4, 8],  # Depth multiplier for the fourth DepthwiseConv1D
    'activationDCNN4': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the fourth DepthwiseConv1D

    'pool_size2': [2, 3, 4],  # Pool sizes for the second MaxPooling1D
    'strides2': [1, 2, 3],  # Stride values for the second MaxPooling1D

    'kernel_size5': [3, 5, 7, 9],  # Kernel sizes for the fifth DepthwiseConv1D
    'padding5': ['same', 'valid'],  # Padding options for the fifth DepthwiseConv1D
    'depth_multiplier5': [1, 2, 4, 8],  # Depth multiplier for the fifth DepthwiseConv1D
    'activationDCNN5': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the fifth DepthwiseConv1D

    'kernel_size6': [3, 5, 7, 9],  # Kernel sizes for the sixth DepthwiseConv1D
    'padding6': ['same', 'valid'],  # Padding options for the sixth DepthwiseConv1D
    'depth_multiplier6': [1, 2, 4, 8],  # Depth multiplier for the sixth DepthwiseConv1D
    'activationDCNN6': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the sixth DepthwiseConv1D

    'pool_size3': [2, 3, 4],  # Pool sizes for the third MaxPooling1D
    'strides3': [1, 2, 3],  # Stride values for the third MaxPooling1D

    'separable_filters': [32, 64, 128, 256],  # Filter counts for SeparableConv1D
    'separable_kernel_size': [3, 5, 7],  # Kernel sizes for SeparableConv1D
    'separable_depth_multiplier': [1, 2, 4],  # Depth multiplier for SeparableConv1D
    'activationCNN1': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for SeparableConv1D

    'conv_filters': [16, 32, 64, 128],  # Filter counts for Conv1D
    'conv_kernel_size': [3, 5, 7],  # Kernel sizes for Conv1D
    'conv_padding': ['same', 'valid'],  # Padding options for Conv1D

    'final_pool_size': [2, 3, 5],  # Pool sizes for the final MaxPooling1D
    'final_pool_strides': [1, 2, 3],  # Stride values for the final MaxPooling1D

    'spatial_dropout2': (float, 0.01, 0.5),  # Range for the second spatial dropout rate

    'dense_units1': [32, 64, 128, 256],  # Units for the first Dense layer
    'activationDense1': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the first Dense layer
    'dense_units2': [16, 32, 64, 128],  # Units for the second Dense layer
    'activationDense2': ['relu', 'selu', 'elu', 'swish'],  # Activation functions for the second Dense layer
    'dropout_rate': (float, 0.01, 0.5),  # Range for dropout rate

    'output_units': [1, 2, 3, 10],  # Units for the output layer
    'activationDense3': ['sigmoid', 'softmax'],  # Activation functions for the output layer
}



def transformer_model(input_shape, params={}):
    """
    Builds a transformer model for 1D data.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled transformer model.
    """
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward block
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    inputs = Input(shape=input_shape)
    x = inputs

    # Stacking Transformer blocks
    for _ in range(params.get('num_transformer_blocks', 1)):
        x = transformer_encoder(
            x,
            head_size=params.get('head_size', 16),
            num_heads=params.get('num_heads', 2),
            ff_dim=params.get('ff_dim', 8),
            dropout=params.get('dropout', 0.05),
        )

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # Fully connected layers after transformer blocks
    for dim in params.get('mlp_units', [32, 8]):
        x = Dense(dim, activation="relu")(x)
        x = Dropout(params.get('mlp_dropout', 0.1))(x)

    outputs = Dense(units=1, activation="sigmoid")(x)
    return Model(inputs, outputs)


@framework('tensorflow')
def transformer_VG(input_shape, params={}):
    return transformer_model(input_shape, {
                            'head_size': params.get('head_size', 16),
                            'num_heads': params.get('num_heads', 32),
                            'ff_dim': params.get('ff_dim', 8),
                            'num_transformer_blocks': params.get('num_transformer_blocks', 1),
                            'mlp_units': params.get('mlp_units', [32, 8]),
                            'dropout': params.get('dropout', 0.05),
                            'mlp_dropout': params.get('mlp_dropout', 0.1),
                        })


@framework('tensorflow')
def transformer(input_shape, params={}):
    return transformer_model(input_shape, {
                            'head_size': params.get('head_size', 8),
                            'num_heads': params.get('num_heads', 2),
                            'ff_dim': params.get('ff_dim', 4),
                            'num_transformer_blocks': params.get('num_transformer_blocks', 1),
                            'mlp_units': params.get('mlp_units', [8]),
                            'dropout': params.get('dropout', 0.05),
                            'mlp_dropout': params.get('mlp_dropout', 0.1),
                        })
    
    
@framework('tensorflow')
def decon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a CNN model with depthwise and separable convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        64,
        kernel_size=3,
        depth_multiplier=1,
        padding="same",
        activation="relu"
    ))
    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(0.1))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


def decon_Sep_classification(input_shape, num_classes=2, params={}):
    """
    Builds a CNN model with separable convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(SeparableConv1D(
        filters=params.get('filters1', 64),
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 2),
        depth_multiplier=params.get('depth_multiplier1', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters2', 64),
        kernel_size=params.get('kernel_size2', 3),
        strides=params.get('strides2', 2),
        depth_multiplier=params.get('depth_multiplier2', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters3', 64),
        kernel_size=params.get('kernel_size3', 3),
        depth_multiplier=params.get('depth_multiplier3', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters4', 64),
        kernel_size=params.get('kernel_size4', 3),
        depth_multiplier=params.get('depth_multiplier4', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(Conv1D(
        filters=params.get('filters5', 32),
        kernel_size=params.get('kernel_size5', 5),
        strides=params.get('strides5', 2),
        padding="same",
        activation="relu"
    ))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=params.get('dense_units', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    if num_classes >= 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def nicon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model with depthwise convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation="sigmoid"))
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model with depthwise convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=params.get('kernel_size1', 15), strides=params.get('strides1', 5), activation=params.get('activation1', "selu")))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=params.get('kernel_size2', 21), strides=params.get('strides2', 3), activation=params.get('activation2', "relu")))
    model.add(BatchNormalization() if params.get('normalization_method1', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=params.get('kernel_size3', 5), strides=params.get('strides3', 3), activation=params.get('activation3', "elu")))
    model.add(BatchNormalization() if params.get('normalization_method2', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation=params.get('dense_activation', "sigmoid")))
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def nicon_VG_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters1', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters3', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters4', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Flatten())
    model.add(Dense(units=params.get('dense_units1', 1024), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units2', 1024), activation="relu"))
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def decon_layer_classification(input_shape, num_classes=2, params={}):
    """
    Builds a model using depthwise separable convolutions and layer normalization for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size2', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size3', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size4', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size5', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size6', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.1)))
    model.add(Flatten())

    # Fully Connected layers
    model.add(Dense(units=params.get('dense_units1', 128), activation="relu"))
    model.add(Dense(units=params.get('dense_units2', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))

    # Output layer
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))

    return model


def transformer_model_classification(input_shape, num_classes=2, params={}):
    """
    Builds a transformer model for 1D data classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled transformer classification model.
    """
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward block
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    inputs = Input(shape=input_shape)
    x = inputs

    # Stacking Transformer blocks
    for _ in range(params.get('num_transformer_blocks', 1)):
        x = transformer_encoder(
            x,
            head_size=params.get('head_size', 16),
            num_heads=params.get('num_heads', 2),
            ff_dim=params.get('ff_dim', 8),
            dropout=params.get('dropout', 0.05),
        )

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # Fully connected layers after transformer blocks
    for dim in params.get('mlp_units', [32, 8]):
        x = Dense(dim, activation="relu")(x)
        x = Dropout(params.get('mlp_dropout', 0.1))(x)

    if num_classes == 2:
        outputs = Dense(units=1, activation="sigmoid")(x)
    else:
        outputs = Dense(units=num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


@framework('tensorflow')
def transformer_VG_classification(input_shape, num_classes=2, params={}):
    return transformer_model_classification(input_shape, num_classes, {
        'head_size': params.get('head_size', 16),
        'num_heads': params.get('num_heads', 32),
        'ff_dim': params.get('ff_dim', 8),
        'num_transformer_blocks': params.get('num_transformer_blocks', 1),
        'mlp_units': params.get('mlp_units', [32, 8]),
        'dropout': params.get('dropout', 0.05),
        'mlp_dropout': params.get('mlp_dropout', 0.1),
    })


@framework('tensorflow')
def transformer_classification(input_shape, num_classes=2, params={}):
    return transformer_model_classification(input_shape, num_classes, {
        'head_size': params.get('head_size', 8),
        'num_heads': params.get('num_heads', 2),
        'ff_dim': params.get('ff_dim', 4),
        'num_transformer_blocks': params.get('num_transformer_blocks', 1),
        'mlp_units': params.get('mlp_units', [8]),
        'dropout': params.get('dropout', 0.05),
        'mlp_dropout': params.get('mlp_dropout', 0.1),
    })


# def build_model(input_shape, params, task_type='regression', num_classes=1):
#     # ... build your model layers ...

#     # Adjust the output layer based on the task type
#     if task_type == 'classification':
#         if num_classes == 2:
#             activation = 'sigmoid'
#             units = 1
#         else:
#             activation = 'softmax'
#             units = num_classes
#     else:
#         activation = 'linear'
#         units = 1

#     model.add(Dense(units=units, activation=activation))
#     return model
