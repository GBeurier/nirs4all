import tensorflow as tf


def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(x, model_width, kernel, strides):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width, kernel, strides=strides, padding='same')(x)  # Stride = 2, Kernel Size = 4
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def Feature_Extraction_Block(inputs, model_width, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1])(latent)
    latent = tf.keras.layers.Reshape((shape[1], model_width))(latent)

    return latent


def Attention_Block(skip_connection, gating_signal, num_filters):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv1D(num_filters, 1, strides=2)(skip_connection)
    conv1x1_1 = tf.keras.layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = tf.keras.layers.Conv1D(num_filters, 1, strides=1)(gating_signal)
    conv1x1_2 = tf.keras.layers.BatchNormalization()(conv1x1_2)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv1D(1, 1, strides=1)(conv1_2)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler = trans_conv1D(conv1_2, num_filters, 4, 1)
    out = skip_connection*resampler

    return out


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3, 2)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)

    return conv


def residual_block(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv_1D_Block(inputs, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for _ in range(n_blocks):
        out = residual_block(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


def stem_bottleneck(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)

    return pool


def residual_block_bottleneck(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 4, 1, 1)
    #
    conv = Conv_1D_Block(inputs, num_filters, 1, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters * 4, 1, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation('relu')(conv)

    return out


def residual_group_bottleneck(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


def encoder18(inputs, num_filters):
    # Construct the Learner
    x1 = residual_group(inputs, num_filters, 2)          # First Residual Block Group of 64 filters
    x2 = residual_group(x1, num_filters * 2, 1)           # Second Residual Block Group of 128 filters
    x3 = residual_group(x2, num_filters * 4, 1)           # Third Residual Block Group of 256 filters
    x4 = residual_group(x3, num_filters * 8, 1, False)  # Fourth Residual Block Group of 512 filters

    return x1, x2, x3, x4


def encoder34(inputs, num_filters):
    # Construct the Learner
    x1 = residual_group(inputs, num_filters, 3)          # First Residual Block Group of 64 filters
    x2 = residual_group(x1, num_filters * 2, 3)           # Second Residual Block Group of 128 filters
    x3 = residual_group(x2, num_filters * 4, 5)           # Third Residual Block Group of 256 filters
    x4 = residual_group(x3, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return x1, x2, x3, x4


def encoder50(inputs, num_filters):
    # Construct the Learner
    x1 = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x2 = residual_group_bottleneck(x1, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x3 = residual_group_bottleneck(x2, num_filters * 4, 5)   # Third Residual Block Group of 256 filters
    x4 = residual_group_bottleneck(x3, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return x1, x2, x3, x4


def encoder101(inputs, num_filters):
    # Construct the Learner
    x1 = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x2 = residual_group_bottleneck(x1, num_filters * 2, 3)   # Second Residual Block Group of 128 filters
    x3 = residual_group_bottleneck(x2, num_filters * 4, 22)  # Third Residual Block Group of 256 filters
    x4 = residual_group_bottleneck(x3, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return x1, x2, x3, x4


def encoder152(inputs, num_filters):
    # Construct the Learner
    x1 = residual_group_bottleneck(inputs, num_filters, 3)  # First Residual Block Group of 64 filters
    x2 = residual_group_bottleneck(x1, num_filters * 2, 7)   # Second Residual Block Group of 128 filters
    x3 = residual_group_bottleneck(x2, num_filters * 4, 35)  # Third Residual Block Group of 256 filters
    x4 = residual_group_bottleneck(x3, num_filters * 8, 2, False)  # Fourth Residual Block Group of 512 filters

    return x1, x2, x3, x4


def decoder_unit(inputs, num_filters):
    # Decoder Unit
    conv = Conv_1D_Block(inputs, num_filters, 1, 1)
    deconv = trans_conv1D(conv, num_filters, 4, 2)
    conv = Conv_1D_Block(deconv, num_filters, 1, 1)

    return conv


def decoder_block(x1, x2, x3, x4, num_filters, D_S, A_G):
    # Decoder Block
    levels = []
    if D_S == 1:
        level = tf.keras.layers.Conv1D(1, 1, name=f'level4')(x4)
        levels.append(level)
    decode1 = decoder_unit(x4, num_filters * 8)
    shape3 = x3.shape
    x3 = Conv_1D_Block(x3, num_filters * 8, 1, shape3[1] + 1)
    if A_G == 1:
        x3 = Attention_Block(x3, decode1, num_filters * 8)
    decode1 = tf.keras.layers.concatenate([decode1, x3], axis=-1)
    if D_S == 1:
        level = tf.keras.layers.Conv1D(1, 1, name=f'level3')(decode1)
        levels.append(level)
    decode2 = decoder_unit(decode1, num_filters * 4)
    shape2 = x2.shape
    x2 = Conv_1D_Block(x2, num_filters * 4, 1, shape2[1] + 1)
    if A_G == 1:
        x2 = Attention_Block(x2, decode2, num_filters * 4)
    decode2 = tf.keras.layers.concatenate([decode2, x2], axis=-1)
    if D_S == 1:
        level = tf.keras.layers.Conv1D(1, 1, name=f'level2')(decode2)
        levels.append(level)
    decode3 = decoder_unit(decode2, num_filters * 2)
    shape1 = x1.shape
    x1 = Conv_1D_Block(x1, num_filters * 2, 1, shape1[1]+1)
    if A_G == 1:
        x1 = Attention_Block(x1, decode3, num_filters * 2)
    decode3 = tf.keras.layers.concatenate([decode3, x1], axis=-1)
    if D_S == 1:
        level = tf.keras.layers.Conv1D(1, 1, name=f'level1')(decode3)
        levels.append(level)
    decode4 = decoder_unit(decode3, num_filters)
    deconv5 = trans_conv1D(decode4, num_filters, 3, 2)
    deconv6 = Conv_1D_Block(deconv5, num_filters, 3, 1)
    out = Conv_1D_Block(deconv6, num_filters, 2, 1)
    if D_S == 1:
        level = tf.keras.layers.Conv1D(1, 1, name=f'level0')(decode4)
        levels.append(level)

    return out, levels


class AlbUNet:
    def __init__(self, length, num_channel, num_filters, ds=0, ae=0, ag=0, problem_type='Regression',
                 output_nums=1, pooling='avg', feature_number=1024, dropout_rate=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # alpha: This Parameter is only for MultiResUNet, default value is 1
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.pooling = pooling
        self.feature_number = feature_number
        self.dropout_rate = dropout_rate

    def MLP(self, x):
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def AlbUNet18(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # Input

        # Encoder
        stem_ = stem(inputs, self.num_filters)
        x1, x2, x3, x4 = encoder18(stem_, self.num_filters)

        # Feature Extraction
        if self.A_E == 1:
            x4 = Feature_Extraction_Block(x4, self.num_filters, self.feature_number)

        # Decoder
        decoder_output, levels = decoder_block(x1, x2, x3, x4, self.num_filters, self.D_S, self.A_G)
        outputs = self.MLP(decoder_output)

        # Instantiate the Model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def AlbUNet34(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # Input

        # Encoder
        stem_ = stem(inputs, self.num_filters)
        x1, x2, x3, x4 = encoder34(stem_, self.num_filters)

        # Feature Extraction
        if self.A_E == 1:
            x4 = Feature_Extraction_Block(x4, self.num_filters, self.feature_number)

        # Decoder
        decoder_output, levels = decoder_block(x1, x2, x3, x4, self.num_filters, self.D_S, self.A_G)
        outputs = self.MLP(decoder_output)

        # Instantiate the Model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def AlbUNet50(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # Input

        # Encoder
        stem_ = stem(inputs, self.num_filters)
        x1, x2, x3, x4 = encoder50(stem_, self.num_filters)

        # Feature Extraction
        if self.A_E == 1:
            x4 = Feature_Extraction_Block(x4, self.num_filters, self.feature_number)

        # Decoder
        decoder_output, levels = decoder_block(x1, x2, x3, x4, self.num_filters, self.D_S, self.A_G)
        outputs = self.MLP(decoder_output)

        # Instantiate the Model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def AlbUNet101(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # Input

        # Encoder
        stem_ = stem(inputs, self.num_filters)
        x1, x2, x3, x4 = encoder101(stem_, self.num_filters)

        # Feature Extraction
        if self.A_E == 1:
            x4 = Feature_Extraction_Block(x4, self.num_filters, self.feature_number)

        # Decoder
        decoder_output, levels = decoder_block(x1, x2, x3, x4, self.num_filters, self.D_S, self.A_G)
        outputs = self.MLP(decoder_output)

        # Instantiate the Model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def AlbUNet152(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # Input

        # Encoder
        stem_ = stem(inputs, self.num_filters)
        x1, x2, x3, x4 = encoder152(stem_, self.num_filters)

        # Feature Extraction
        if self.A_E == 1:
            x4 = Feature_Extraction_Block(x4, self.num_filters, self.feature_number)

        # Decoder
        decoder_output, levels = decoder_block(x1, x2, x3, x4, self.num_filters, self.D_S, self.A_G)
        outputs = self.MLP(decoder_output)

        # Instantiate the Model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model


if __name__ == '__main__':
    # Configurations
    num_channel = 1
    length = 1024
    model_width = 64
    D_S = 1
    A_E = 0
    A_G = 1
    feature_number = 1024
    model_name = 'AlbUNet152'
    # Build model for AlbUNet
    Model = AlbUNet(length, num_channel, model_width, ds=D_S, ae=A_E, ag=A_G, problem_type='Regression', output_nums=1, pooling='avg', dropout_rate=False).AlbUNet152()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
