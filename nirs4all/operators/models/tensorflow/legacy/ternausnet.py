# Import Necessary Libraries
import tensorflow as tf

def Conv_Block(inputs, model_width, kernel, multiplier=1, bottleneck=False):
    # 1D Convolutional Block
    if bottleneck:
        # Bottleneck design: use 1x1 conv to reduce channels before 3x3 conv
        x = tf.keras.layers.Conv1D(model_width * multiplier // 4, 1, padding='same', kernel_initializer="he_normal")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same', kernel_initializer="he_normal")(x)
    else:
        # Standard convolution
        x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same', kernel_initializer="he_normal")(inputs)
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the Keras Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = tf.keras.layers.UpSampling1D(size=2)(inputs)

    return up


def trans_conv1D(inputs, model_width, multiplier, strides):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, 4, strides=strides, padding='same')(inputs)  # Stride = 2, Kernel Size = 4
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


def Attention_Block(skip_connection, gating_signal, num_filters, multiplier):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv1D(num_filters * multiplier, 1, strides=2)(skip_connection)
    conv1x1_1 = tf.keras.layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = tf.keras.layers.Conv1D(num_filters * multiplier, 1, strides=1)(gating_signal)
    conv1x1_2 = tf.keras.layers.BatchNormalization()(conv1x1_2)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv1D(1, 1, strides=1)(conv1_2)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler = upConv_Block(conv1_2)
    out = skip_connection * resampler

    return out


class TernausNet:
    def __init__(self, length, num_channel, model_width, ds=0, ae=0, ag=0, problem_type='Regression',
                 output_nums=1, feature_number=1024, is_transconv=True):
        # length: Input Signal Length
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        self.length = length
        self.num_channel = num_channel
        self.model_width = model_width
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.feature_number = feature_number
        self.is_transconv = is_transconv

    def TernausNet11(self):
        """TernausNet11 Model Design - UNet with VGG11 Backend"""
        if self.length == 0 or self.model_width == 0 or self.num_channel == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, 3, 2 ** 0)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv0"] = conv
        conv = Conv_Block(pool, self.model_width, 3, 2 ** 1)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv1"] = conv

        for i in range(2, 4):
            conv = Conv_Block(pool, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
            convs["conv%s" % i] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv4"] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
            conv = Conv_Block(latent, self.model_width, 3, 2 ** 3)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Decoding
        convs_list = list(convs.values())
        deconv = conv
        for j in range(0, 5):
            skip_connection = convs_list[4 - j]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[4 - j], deconv, self.model_width, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            if self.is_transconv:
                deconv = Concat_Block(trans_conv1D(deconv, self.model_width, 2 ** (4 - j), 2), skip_connection)
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), skip_connection)
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{4 - j}')(deconv)
                levels.append(level)

        deconv = Conv_Block(deconv, self.model_width, 3, 2 ** 0)

        # Output
        outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def TernausNet13(self):
        """TernausNet13 Model Design - UNet with VGG13 Backend"""
        if self.length == 0 or self.model_width == 0 or self.num_channel == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, 3, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 0)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv0"] = conv
        conv = Conv_Block(pool, self.model_width, 3, 2 ** 1)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 1)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv1"] = conv

        for i in range(2, 4):
            conv = Conv_Block(pool, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
            convs["conv%s" % i] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv4"] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
            conv = Conv_Block(latent, self.model_width, 3, 2 ** 3)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Decoding
        convs_list = list(convs.values())
        deconv = conv
        for j in range(0, 5):
            skip_connection = convs_list[4 - j]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[4 - j], deconv, self.model_width, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            if self.is_transconv:
                deconv = Concat_Block(trans_conv1D(deconv, self.model_width, 2 ** (4 - j), 2), skip_connection)
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), skip_connection)
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{4 - j}')(deconv)
                levels.append(level)

        deconv = Conv_Block(deconv, self.model_width, 3, 2 ** 0)

        # Output
        outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def TernausNet16(self):
        """TernausNet16 Model Design - UNet with VGG16 Backend"""
        if self.length == 0 or self.model_width == 0 or self.num_channel == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, 3, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 0)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv0"] = conv
        conv = Conv_Block(pool, self.model_width, 3, 2 ** 1)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 1)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv1"] = conv

        for i in range(2, 4):
            conv = Conv_Block(pool, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 1, 2 ** i)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
            convs["conv%s" % i] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 1, 2 ** 3)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv4"] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
            conv = Conv_Block(latent, self.model_width, 3, 2 ** 3)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Decoding
        convs_list = list(convs.values())

        deconv = conv
        for j in range(0, 5):
            skip_connection = convs_list[4 - j]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[4 - j], deconv, self.model_width, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            if self.is_transconv:
                deconv = Concat_Block(trans_conv1D(deconv, self.model_width, 2 ** (4 - j), 2), skip_connection)
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), skip_connection)
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{4 - j}')(deconv)
                levels.append(level)

        deconv = Conv_Block(deconv, self.model_width, 3, 2 ** 0)

        # Output
        outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model

    def TernausNet19(self):
        """TernausNet19 Model Design - UNet with VGG19 Backend"""
        if self.length == 0 or self.model_width == 0 or self.num_channel == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        conv = Conv_Block(inputs, self.model_width, 3, 2 ** 0)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 0)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv0"] = conv
        conv = Conv_Block(pool, self.model_width, 3, 2 ** 1)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 1)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv1"] = conv

        for i in range(2, 4):
            conv = Conv_Block(pool, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** i)
            pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
            convs["conv%s" % i] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(conv)
        convs["conv4"] = conv

        conv = Conv_Block(pool, self.model_width, 3, 2 ** 3)
        conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 1:
            latent = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
            conv = Conv_Block(latent, self.model_width, 3, 2 ** 3)
            conv = Conv_Block(conv, self.model_width, 3, 2 ** 3)

        # Decoding
        convs_list = list(convs.values())

        deconv = conv
        for j in range(0, 5):
            skip_connection = convs_list[4 - j]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[4 - j], deconv, self.model_width, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            deconv = Conv_Block(deconv, self.model_width, 3, 2 ** (4 - j))
            if self.is_transconv:
                deconv = Concat_Block(trans_conv1D(deconv, self.model_width, 2 ** (4 - j), 2), skip_connection)
            elif not self.is_transconv:
                deconv = Concat_Block(upConv_Block(deconv), skip_connection)
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{4 - j}')(deconv)
                levels.append(level)

        deconv = Conv_Block(deconv, self.model_width, 3, 2 ** 0)

        # Output
        outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)

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
    A_E = 1
    A_G = 1
    feature_number = 1024
    model_name = 'TernausNet16'
    # Build model for TernausNet
    Model = TernausNet(length, num_channel, model_width, problem_type='Regression', output_nums=1, ds=D_S, ae=A_E, ag=A_G, is_transconv=True).TernausNet16()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
