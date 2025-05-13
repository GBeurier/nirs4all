import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling1D,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    ConvLSTM1D,
    Dense,
    DepthwiseConv1D,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPool1D,
    GRU,
    Input,
    Lambda,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
    MaxPool1D,
    MultiHeadAttention,
    Multiply,
    Permute,
    ReLU,
    SeparableConv1D,
    SpatialDropout1D,
    UpSampling1D,
)

from keras.models import Model, Sequential
from nirs4all.core.utils import framework
from nirs4all.presets.legacy.Inception_1DCNN import Inception
from nirs4all.presets.legacy.ResNet_v2_1DCNN import ResNetv2
from nirs4all.presets.legacy.SE_ResNet_1DCNN import SEResNet
from nirs4all.presets.legacy.VGG_1DCNN import VGG


@framework('tensorflow')
def UNet_NIRS(input_shape, params):
    """
    Builds a U-Net model for NIRS data.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled U-Net model.
    """
    length = input_shape[0]
    num_channel = input_shape[1]
    model_width = params.get('model_width', 8)
    problem_type = params.get('problem_type', 'Regression')
    output_nums = params.get('output_nums', 1)
    dropout_rate = params.get('dropout_rate', False)

    # Assuming VGG class is imported or defined elsewhere
    # Replace VGG with appropriate model or define it here
    model = VGG(
        length,
        num_channel,
        model_width,
        problem_type=problem_type,
        output_nums=output_nums,
        dropout_rate=dropout_rate,
    ).VGG11()
    return model


@framework('tensorflow')
def VGG_1D(input_shape, params):
    """
    Builds a VGG-like 1D CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled VGG-like model.
    """
    def vgg_block(layer_in, n_filters, n_conv):
        for _ in range(n_conv):
            layer_in = Conv1D(
                filters=n_filters,
                kernel_size=params.get('kernel_size', 3),
                padding="same",
                activation="relu"
            )(layer_in)
        layer_in = MaxPooling1D(pool_size=2, strides=2)(layer_in)
        return layer_in

    visible = Input(shape=input_shape)
    layer = vgg_block(visible, params.get('block1_filters', 64), params.get('block1_convs', 2))
    layer = vgg_block(layer, params.get('block2_filters', 128), params.get('block2_convs', 2))
    layer = vgg_block(layer, params.get('block3_filters', 256), params.get('block3_convs', 2))
    layer = Flatten()(layer)
    layer = Dense(units=params.get('dense_units', 16), activation="sigmoid")(layer)
    layer = Dense(units=1, activation="linear")(layer)
    model = Model(inputs=visible, outputs=layer)
    return model


@framework('tensorflow')
def CONV_LSTM(input_shape, params):
    """
    Builds a model combining convolutional and LSTM layers.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Convolutional Path
    conv_filters = params.get('conv_filters', 64)
    conv_kernel_size = params.get('conv_kernel_size', 3)
    conv_depth_multiplier = params.get('conv_depth_multiplier', 64)
    x1 = SeparableConv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        depth_multiplier=conv_depth_multiplier,
        padding="same",
        activation="relu",
    )(x)
    x1 = MaxPooling1D()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)

    # Attention Path
    x2 = MultiHeadAttention(
        key_dim=params.get('attention_key_dim', 64),
        num_heads=params.get('attention_num_heads', 8),
        dropout=0.1
    )(x, x)
    x2 = MaxPooling1D()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(32, 3, strides=2, padding="same")(x2)
    x2 = Flatten()(x2)

    # GRU Path
    x3 = Bidirectional(GRU(params.get('gru_units', 128)))(x)
    x3 = BatchNormalization()(x3)

    # LSTM Path
    x4 = Bidirectional(LSTM(params.get('lstm_units', 128)))(x)
    x4 = BatchNormalization()(x4)

    # Concatenate Paths
    x = Concatenate()([x1, x2, x3, x4])
    x = BatchNormalization()(x)

    # Fully Connected Layers
    x = Dense(units=params.get('fc_units1', 64), activation="relu")(x)
    x = Dense(units=params.get('fc_units2', 16), activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1, activation="sigmoid")(x)
    return Model(inputs, outputs)


@framework('tensorflow')
def UNET(input_shape, params):
    """
    Builds a U-Net architecture for regression.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled U-Net model.
    """
    layer_n = params.get('layer_n', 64)
    kernel_size = params.get('kernel_size', 7)
    depth = params.get('depth', 2)

    def cbr(x, out_layer, kernel, stride, dilation):
        x = Conv1D(
            out_layer,
            kernel_size=kernel,
            dilation_rate=dilation,
            strides=stride,
            padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def se_block(x_in, layer_n):
        x = GlobalAveragePooling1D()(x_in)
        x = Dense(layer_n // 8, activation="relu")(x)
        x = Dense(layer_n, activation="sigmoid")(x)
        x_out = Multiply()([x_in, x])
        return x_out

    def resblock(x_in, layer_n, kernel, dilation, use_se=True):
        x = cbr(x_in, layer_n, kernel, 1, dilation)
        x = cbr(x, layer_n, kernel, 1, dilation)
        if use_se:
            x = se_block(x, layer_n)
        x = Add()([x_in, x])
        return x

    input_layer = Input(input_shape)
    input_layer_1 = AveragePooling1D(5)(input_layer)
    input_layer_2 = AveragePooling1D(25)(input_layer)

    # Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for _ in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    # out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 5, 1)
    for _ in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    # out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, kernel_size, 5, 1)
    for _ in range(depth):
        x = resblock(x, layer_n * 3, kernel_size, 1)
    # out_2 = x

    x = Concatenate()([x, input_layer_2])
    x = cbr(x, layer_n * 4, kernel_size, 5, 1)
    for _ in range(depth):
        x = resblock(x, layer_n * 4, kernel_size, 1)

    # Regression Output
    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: 12 * x)(x)
    x = Flatten()(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(input_layer, out)
    return model


@framework('tensorflow')
def bard(input_shape, params):
    """
    Builds a model combining convolutional and LSTM layers.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    input_layer = Input(shape=input_shape)
    conv1 = SeparableConv1D(
        filters=params.get('conv_filters1', 64),
        kernel_size=params.get('conv_kernel_size1', 3),
        padding='same'
    )(input_layer)
    conv2 = SeparableConv1D(
        filters=params.get('conv_filters2', 16),
        kernel_size=params.get('conv_kernel_size2', 5),
        padding='same'
    )(conv1)
    batch_norm1 = BatchNormalization()(conv2)
    activation1 = ReLU()(batch_norm1)
    flattened_conv = Flatten()(activation1)

    lstm = Bidirectional(
        LSTM(
            units=params.get('lstm_units1', 64),
            return_sequences=True
        )
    )(input_layer)
    lstm2 = LSTM(
        units=params.get('lstm_units2', 32),
        return_sequences=False
    )(lstm)
    batch_norm2 = BatchNormalization()(lstm2)
    flattened_lstm = Flatten()(batch_norm2)

    concatenated = Concatenate()([flattened_conv, flattened_lstm])
    dense = Dense(params.get('dense_units1', 128))(concatenated)
    dense2 = Dense(params.get('dense_units2', 8))(dense)
    predictions = Dense(1, activation='linear')(dense2)
    model = Model(input_layer, predictions)
    return model


@framework('tensorflow')
def XCeption1D(input_shape, params):
    """
    Builds an Xception-like 1D CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled Xception-like model.
    """
    def xception_entry_flow(inputs):
        x = DepthwiseConv1D(
            kernel_size=params.get('entry_kernel_size1', 3),
            strides=params.get('entry_strides1', 2),
            depth_multiplier=params.get('entry_depth_multiplier1', 2),
            padding='same'
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = DepthwiseConv1D(
            kernel_size=params.get('entry_kernel_size2', 3),
            strides=params.get('entry_strides2', 2),
            depth_multiplier=params.get('entry_depth_multiplier2', 2),
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        previous_block_activation = x

        for size in params.get('entry_sizes', [128, 256, 728]):
            x = Activation("relu")(x)
            x = SeparableConv1D(size, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(size, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPool1D(3, strides=2, padding="same")(x)

            residual = Conv1D(size, 1, strides=2, padding="same")(previous_block_activation)

            x = Add()([x, residual])
            previous_block_activation = x

        return x

    def xception_middle_flow(x, num_blocks=8):
        previous_block_activation = x
        for _ in range(num_blocks):
            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Add()([x, previous_block_activation])
            previous_block_activation = x

        return x

    def xception_exit_flow(x):
        previous_block_activation = x

        x = Activation("relu")(x)
        x = SeparableConv1D(728, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv1D(1024, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPool1D(3, strides=2, padding="same")(x)

        residual = Conv1D(1024, 1, strides=2, padding="same")(previous_block_activation)
        x = Add()([x, residual])

        x = Activation("relu")(x)
        x = SeparableConv1D(728, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv1D(1024, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(1, activation="sigmoid")(x)

        return x

    inputs = Input(shape=input_shape)
    x = xception_entry_flow(inputs)
    x = xception_middle_flow(x, num_blocks=params.get('middle_num_blocks', 8))
    outputs = xception_exit_flow(x)
    return Model(inputs, outputs)


@framework('tensorflow')
def MLP(input_shape, params):
    """
    Builds a simple Multi-Layer Perceptron model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled MLP model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units1', 1024), activation="relu"))
    model.add(Dense(units=params.get('dense_units2', 128), activation="relu"))
    model.add(Dense(units=params.get('dense_units3', 8), activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


@framework('tensorflow')
def Custom_VG_Residuals2(input_shape, params):
    """
    Builds a custom residual CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    inputs = Input(shape=input_shape)
    x = SpatialDropout1D(params.get('spatial_dropout', 0.2))(inputs)

    # Branch 1
    x1 = DepthwiseConv1D(
        kernel_size=params.get('branch1_kernel_size1', 3),
        padding="same",
        depth_multiplier=params.get('branch1_depth_multiplier1', 8),
        activation="relu",
    )(x)
    x1 = DepthwiseConv1D(
        kernel_size=params.get('branch1_kernel_size2', 3),
        padding="same",
        depth_multiplier=params.get('branch1_depth_multiplier2', 2),
        activation="sigmoid",
    )(x1)

    # Branch 2
    x2 = DepthwiseConv1D(
        kernel_size=params.get('branch2_kernel_size1', 7),
        padding="same",
        depth_multiplier=params.get('branch2_depth_multiplier1', 8),
        activation="relu",
    )(x)
    x2 = DepthwiseConv1D(
        kernel_size=params.get('branch2_kernel_size2', 7),
        padding="same",
        depth_multiplier=params.get('branch2_depth_multiplier2', 2),
        activation="sigmoid",
    )(x2)

    # Branch 3
    x3 = DepthwiseConv1D(
        kernel_size=params.get('branch3_kernel_size1', 15),
        padding="same",
        depth_multiplier=params.get('branch3_depth_multiplier1', 8),
        activation="relu",
    )(x)
    x3 = DepthwiseConv1D(
        kernel_size=params.get('branch3_kernel_size2', 15),
        padding="same",
        depth_multiplier=params.get('branch3_depth_multiplier2', 2),
        activation="sigmoid",
    )(x3)

    # Concatenate branches
    x = Concatenate(axis=2)([x1, x2, x3])
    x = BatchNormalization()(x)

    # Convolutional Layers
    x = Conv1D(
        filters=params.get('conv_filters1', 64),
        kernel_size=params.get('conv_kernel_size1', 7),
        strides=params.get('conv_strides1', 5),
        activation="relu",
    )(x)
    x = Conv1D(
        filters=params.get('conv_filters2', 16),
        kernel_size=params.get('conv_kernel_size2', 3),
        strides=params.get('conv_strides2', 3),
        activation="sigmoid",
    )(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)
    x = Dense(params.get('dense_units', 16), activation="sigmoid")(x)
    x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)
    return model


@framework('tensorflow')
def SEResNet_model(input_shape, params):
    """
    Builds a SEResNet model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled SEResNet model.
    """
    length = input_shape[0]
    num_channel = input_shape[-1]
    model_width = params.get('model_width', 16)
    problem_type = params.get('problem_type', 'Regression')
    output_nums = params.get('output_nums', 1)
    reduction_ratio = params.get('reduction_ratio', 4)
    dropout_rate = params.get('dropout_rate', False)
    pooling = params.get('pooling', 'avg')

    # Assuming SEResNet class is defined or imported
    # Replace SEResNet with appropriate model or define it here
    model = SEResNet(
        length,
        num_channel,
        model_width,
        ratio=reduction_ratio,
        problem_type=problem_type,
        output_nums=output_nums,
        pooling=pooling,
        dropout_rate=dropout_rate,
    ).SEResNet101()
    return model


@framework('tensorflow')
def ResNetV2_model(input_shape, params):
    """
    Builds a ResNetV2 model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled ResNetV2 model.
    """
    length = input_shape[0]
    num_channel = input_shape[-1]
    model_width = params.get('model_width', 16)
    problem_type = params.get('problem_type', 'Regression')
    output_nums = params.get('output_nums', 1)
    pooling = params.get('pooling', 'avg')
    dropout_rate = params.get('dropout_rate', False)

    # Assuming ResNetv2 class is defined or imported
    model = ResNetv2(
        length,
        num_channel,
        model_width,
        problem_type=problem_type,
        output_nums=output_nums,
        pooling=pooling,
        dropout_rate=dropout_rate,
    ).ResNet34()
    return model


@framework('tensorflow')
def FFT_Conv(input_shape, params):
    """
    Builds a CNN model with FFT preprocessing.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    inputs = Input(shape=input_shape)
    x = SpatialDropout1D(params.get('spatial_dropout', 0.2))(inputs)
    x = Permute((2, 1))(x)
    x = Lambda(lambda v: tf.cast(tf.signal.fft(tf.cast(v, dtype=tf.complex64)), tf.float32))(x)
    x = Permute((2, 1))(x)
    x = SeparableConv1D(
        filters=params.get('filters1', 64),
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 2),
        depth_multiplier=params.get('depth_multiplier1', 32),
        padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv1D(
        filters=params.get('filters2', 64),
        kernel_size=params.get('kernel_size2', 3),
        strides=params.get('strides2', 2),
        depth_multiplier=params.get('depth_multiplier2', 32),
        padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(
        filters=params.get('filters3', 32),
        kernel_size=params.get('kernel_size3', 5),
        strides=params.get('strides3', 2),
        padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Activation("relu")(x)
    x = Dense(params.get('dense_units1', 128), activation="relu")(x)
    x = Dense(params.get('dense_units2', 32), activation="relu")(x)
    x = Dropout(params.get('dropout_rate', 0.1))(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model


@framework('tensorflow')
def Custom_Residuals(input_shape, params):
    """
    Builds a custom residual model with depthwise convolutions and attention.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    inputs = Input(shape=input_shape)
    x = SpatialDropout1D(params.get('spatial_dropout', 0.2))(inputs)
    x = DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 3),
        depth_multiplier=params.get('depth_multiplier1', 2),
        activation="relu"
    )(x)
    x = DepthwiseConv1D(
        kernel_size=params.get('kernel_size2', 5),
        strides=params.get('strides2', 3),
        activation="relu"
    )(x)
    x = DepthwiseConv1D(
        kernel_size=params.get('kernel_size3', 5),
        strides=params.get('strides3', 3),
        activation="relu"
    )(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = MultiHeadAttention(
        key_dim=params.get('key_dim', 11),
        num_heads=params.get('num_heads', 4),
        dropout=0.1
    )(x, x)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Flatten()(x)
    x = Dense(params.get('dense_units', 16), activation="relu")(x)
    x = Dropout(params.get('dropout_rate', 0.1))(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model


@framework('tensorflow')
def Custom_VG_Residuals(input_shape, params):
    """
    Builds a custom residual model with depthwise convolutions and attention.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled model.
    """
    def block(x, strides):
        x = DepthwiseConv1D(
            kernel_size=params.get('block_kernel_size1', 3),
            strides=strides,
            activation="relu"
        )(x)
        x = DepthwiseConv1D(
            kernel_size=params.get('block_kernel_size2', 5),
            strides=1,
            activation="relu"
        )(x)
        x = DepthwiseConv1D(
            kernel_size=params.get('block_kernel_size3', 5),
            strides=1,
            activation="relu"
        )(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(
            key_dim=params.get('key_dim', 11),
            num_heads=params.get('num_heads', 4),
            dropout=0.1
        )(x, x)
        return x

    inputs = Input(shape=input_shape)
    x = SpatialDropout1D(params.get('spatial_dropout', 0.2))(inputs)
    x = DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 3),
        depth_multiplier=params.get('depth_multiplier1', 2),
        activation="relu"
    )(x)

    x = block(x, 2)
    x = block(x, 2)
    x = block(x, 2)
    x = block(x, 1)
    x = block(x, 1)
    x = block(x, 1)

    x = Conv1D(
        filters=params.get('conv_filters1', 16),
        kernel_size=1,
        activation="relu"
    )(x)
    x = Dropout(params.get('dropout_rate1', 0.2))(x)
    x = Conv1D(
        filters=params.get('conv_filters2', 8),
        kernel_size=params.get('kernel_size2', 8),
        strides=params.get('strides2', 8)
    )(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    x = Flatten()(x)
    x = Dense(params.get('dense_units', 16), activation="relu")(x)
    x = Dropout(params.get('dropout_rate2', 0.1))(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model


@framework('tensorflow')
def inception1D(input_shape, params):
    """
    Builds an Inception-like 1D CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled Inception-like model.
    """
    length = input_shape[1]
    model_width = params.get('model_width', 16)
    num_channel = params.get('num_channel', 1)
    problem_type = params.get('problem_type', 'Regression')
    output_number = params.get('output_number', 1)

    # Assuming Inception class is defined elsewhere, replace with proper Inception implementation
    Regression_Model = Inception(
        length, num_channel, model_width, problem_type=problem_type, output_nums=output_number
    ).Inception_v3()

    return Regression_Model


@framework('tensorflow')
def senseen_origin(input_shape, params):
    """
    Builds a CNN model with custom convolutional blocks.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    def senseen_conv1D_block(model, filters):
        model.add(Conv1D(filters=filters, kernel_size=3, padding="same", activation='relu'))
        model.add(Conv1D(filters=filters, kernel_size=3, padding="same", activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides=3))

    def senseen_depthconv1D_block(model, filters):
        model.add(BatchNormalization())
        model.add(DepthwiseConv1D(kernel_size=3, depth_multiplier=filters, padding="same", activation='swish'))
        model.add(DepthwiseConv1D(kernel_size=3, depth_multiplier=filters, padding="same", activation='swish'))
        model.add(AveragePooling1D(pool_size=4, strides=2))

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.3)))
    senseen_depthconv1D_block(model, params.get('filters1', 4))
    senseen_depthconv1D_block(model, params.get('filters2', 2))
    senseen_depthconv1D_block(model, params.get('filters3', 4))
    senseen_conv1D_block(model, params.get('filters4', 32))
    model.add(Flatten())
    model.add(Dense(units=params.get('dense_units1', 512), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units2', 512), activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model