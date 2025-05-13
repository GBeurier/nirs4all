# VGG 2D-Convolution Architecture in Keras - For both Classification and Regression Problems
"""Reference: [Very Deep Convolutional Networks for Large-Scale Image Recognition] (https://arxiv.org/abs/1409.1556)"""

import tensorflow as tf


def Conv_2D_Block(inputs, model_width, kernel):
    # 2D Convolutional Block with BatchNormalization
    conv = tf.keras.layers.Conv2D(model_width, (kernel, kernel), padding='same')(inputs)
    batch_norm = tf.keras.layers.BatchNormalization()(conv)
    activate = tf.keras.layers.Activation('relu')(batch_norm)

    return activate


class VGG:
    def __init__(self, length, width, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, dropout_rate=False):
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate

    def VGG11(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_2D_Block(inputs, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 2
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 3
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 4
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 5
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def VGG13(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_2D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 2
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 3
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 4
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 5
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def VGG16(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_2D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1]<=2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 2
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 3
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 4
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 5
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def VGG16_v2(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_2D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 2
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 3
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 1)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 4
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 1)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 5
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 1)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def VGG19(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_2D_Block(inputs, self.num_filters * (2 ** 0), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 2
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 3
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 4
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Block 5
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model


if __name__ == '__main__':
    # Configurations
    length = 224  # Length of each Image
    width = 224  # Width of each Image
    model_name = 'VGG19'  # DenseNet Models
    model_width = 16 # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Input Channels in the Model
    problem_type = 'Regression' # Classification or Regression
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    #
    Model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=False).VGG19()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()