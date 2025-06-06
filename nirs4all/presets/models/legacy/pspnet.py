# Import Necessary Libraries
import tensorflow as tf


def Conv_Block(inputs, model_width, kernel, use_batchnorm=True):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width, kernel, padding='same')(inputs)
    if use_batchnorm == True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(inputs, model_width):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width, 2, strides=2, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def PyramidPoolingBlock(inputs, level, conv_filters, pooling_type='avg', use_batchnorm=True):
    # Pyramid Pooling Block
    pool_size = up_size = level

    x = []
    if pooling_type == 'avg':
        x = tf.keras.layers.AveragePooling1D(pool_size, strides=pool_size, padding='same')(inputs)
    elif pooling_type == 'max':
        x = tf.keras.layers.MaxPooling1D(pool_size, strides=pool_size, padding='same')(inputs)
    x = Conv_Block(x, conv_filters, 1, use_batchnorm=use_batchnorm)
    x = tf.keras.layers.UpSampling1D(size=up_size)(x)

    return x


class PSPNet:
    def __init__(self, length, num_channel, model_width, problem_type='Regression', output_nums=1, cardinality=5,
                 is_transconv=True, pooling_type='avg', use_batchnorm=True, dropout=None):
        # length: Input Signal Length
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # is_transconv: Whether to use Transposed Convolution or UpSampling
        # pooling_type: Pooling type Max or Average.
        # dropout: Enable Dropout by mentioning a ratio
        self.length = length
        self.num_channel = num_channel
        self.model_width = model_width
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.cardinality = cardinality
        self.is_transconv = is_transconv
        self.use_batchnorm = use_batchnorm
        self.pooling_type = pooling_type
        self.dropout = dropout

    def PSPNet(self):
        # Variable PSPNet Model Design
        if self.length == 0 or self.model_width == 0 or self.num_channel == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        x = inputs

        # Build Spatial Pyramid
        for i in range(0, self.cardinality):
            x_temp = PyramidPoolingBlock(inputs, 2 ** i, self.model_width, pooling_type=self.pooling_type, use_batchnorm=self.use_batchnorm)
            x = tf.keras.layers.concatenate([x, x_temp], axis=-1)

        x = Conv_Block(x, self.model_width, 1, use_batchnorm=self.use_batchnorm)

        # Model Regularization
        if self.dropout is not None:
            x = tf.keras.layers.SpatialDropout1D(self.dropout)(x)

        # Model Head
        x = Conv_Block(x, self.model_width, 3, use_batchnorm=True)
        out = tf.keras.layers.UpSampling1D(size=2)(x)
        if self.is_transconv:
            out = trans_conv1D(x, self.model_width)

        # Output
        outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(out)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(out)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        return model


if __name__ == '__main__':
    # Configurations
    length = 1024
    model_width = 512
    num_channel = 4
    problem_type = 'Regression'
    output_nums = 1
    cardinality = 5
    is_transconv = True
    model_name = 'PSPNet'
    #
    Model = PSPNet(length, num_channel, model_width, cardinality=cardinality, is_transconv=is_transconv).PSPNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
