import tensorflow as tf
from tensorflow.keras import layers as L

from nirs4all.utils.backend import framework

#####################
## CHATGPT MODELS ##
#####################

@framework('tensorflow')
def resnet_se(input_shape, params: dict = None):
    """1D ResNet with Squeeze-Excitation."""
    if params is None:
        params = {}
    r = params.get('se_ratio', 8)
    ks = params.get('kernel_size', 9)
    filters = params.get('filters', [32, 64, 128])
    pool_stride = params.get('pool_stride', 2)
    dropout = params.get('dropout', 0.2)

    def se_block(x):
        c = x.shape[-1]
        s = L.GlobalAveragePooling1D()(x)
        s = L.Dense(max(c // r, 1), activation='relu')(s)
        s = L.Dense(c, activation='sigmoid')(s)
        s = L.Reshape((1, c))(s)
        return L.Multiply()([x, s])

    def res_block(x, f):
        y = L.Conv1D(f, ks, padding='same')(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        y = L.Conv1D(f, ks, padding='same')(y)
        y = L.BatchNormalization()(y)
        y = se_block(y)
        s = x
        if x.shape[-1] != f:
            s = L.Conv1D(f, 1, padding='same')(x)
        y = L.Add()([s, y])
        y = L.ReLU()(y)
        return y

    inp = L.Input(shape=input_shape)
    x = L.Conv1D(filters[0], 7, padding='same')(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for f in filters:
        x = res_block(x, f)
        x = L.MaxPool1D(pool_size=2, strides=pool_stride)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

@framework('tensorflow')
def inception_time(input_shape, params: dict = None):
    """InceptionTime-style 1D CNN."""
    if params is None:
        params = {}
    f = params.get('filters', 64)
    ks = params.get('kernels', (5, 9, 15))
    blocks = params.get('blocks', 3)
    dropout = params.get('dropout', 0.2)

    def block(x, f):
        b = []
        for k in ks:
            b.append(L.Conv1D(f // 4, k, padding='same', activation='relu')(x))
        p = L.MaxPool1D(3, strides=1, padding='same')(x)
        p = L.Conv1D(f // 4, 1, padding='same', activation='relu')(p)
        y = L.Concatenate()(b + [p])
        y = L.BatchNormalization()(y)
        return y

    inp = L.Input(shape=input_shape)
    x = inp
    for i in range(blocks):
        x = block(x, f if i < blocks - 1 else 2 * f)
    x = L.SpatialDropout1D(dropout)(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

@framework('tensorflow')
def tcn_noncausal(input_shape, params: dict = None):
    """Non-causal TCN with residual dilations."""
    if params is None:
        params = {}
    filters = params.get('filters', [64, 64, 128])
    dilations = params.get('dilations', [1, 2, 4, 8])
    k = params.get('kernel_size', 7)
    p = params.get('spatial_dropout', 0.1)

    def tcn_block(x, f, d):
        y = L.Conv1D(f, k, dilation_rate=d, padding='same')(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        y = L.SpatialDropout1D(p)(y)
        y = L.Conv1D(f, k, dilation_rate=d, padding='same')(y)
        y = L.BatchNormalization()(y)
        s = x if x.shape[-1] == f else L.Conv1D(f, 1, padding='same')(x)
        y = L.Add()([s, y])
        y = L.ReLU()(y)
        return y

    inp = L.Input(shape=input_shape)
    x = inp
    for f in filters:
        for d in dilations:
            x = tcn_block(x, f, d)
        x = L.MaxPool1D(2)(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

@framework('tensorflow')
def conv_transformer(input_shape, params: dict = None):
    """Light Conv → Transformer encoder."""
    if params is None:
        params = {}
    conv_filters = params.get('conv_filters', [64, 128])
    conv_kernel = params.get('conv_kernel', 7)
    conv_stride = params.get('conv_stride', 2)
    heads = params.get('heads', 4)
    dim = params.get('model_dim', 128)
    blocks = params.get('blocks', 2)
    mlp_ratio = params.get('mlp_ratio', 4)

    def encoder(x):
        # Self-attention block
        a = L.MultiHeadAttention(num_heads=heads, key_dim=dim)(x, x)
        x = L.Add()([x, a])
        x = L.LayerNormalization()(x)
        f = L.Dense(mlp_ratio * dim)(x)
        f = L.Activation('gelu')(f)
        f = L.Dense(dim)(f)
        x = L.Add()([x, f])
        x = L.LayerNormalization()(x)
        return x

    inp = L.Input(shape=input_shape)
    x = inp
    for f in conv_filters:
        x = L.Conv1D(f, conv_kernel, strides=conv_stride, padding='same', activation='relu')(x)
    x = L.LayerNormalization()(x)
    # Project to dim if needed
    if x.shape[-1] != dim:
        x = L.Dense(dim)(x)
    for _ in range(blocks):
        x = encoder(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

@framework('tensorflow')
def convmixer1d(input_shape, params: dict = None):
    """Depthwise-Separable ConvMixer for 1D spectra."""
    if params is None:
        params = {}
    dim = params.get('dim', 128)
    depth = params.get('depth', 6)
    patch = params.get('patch_size', 5)
    ds_k = params.get('dw_kernel', 9)
    p = params.get('dropout', 0.1)

    inp = L.Input(shape=input_shape)
    # Patchify / stem
    x = L.Conv1D(dim, patch, strides=patch, padding='same')(inp)
    x = L.Activation('gelu')(x)
    x = L.BatchNormalization()(x)
    for _ in range(depth):
        y = L.DepthwiseConv1D(ds_k, padding='same')(x)
        y = L.Activation('gelu')(y)
        y = L.BatchNormalization()(y)
        x = L.Add()([x, y])
        x = L.Conv1D(dim, 1, padding='same')(x)
        x = L.Activation('gelu')(x)
        x = L.BatchNormalization()(x)
        x = L.SpatialDropout1D(p)(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

@framework('tensorflow')
def cnn_pls_head(input_shape, params: dict = None):
    """CNN trunk with linear head mimicking PLS-style projection."""
    if params is None:
        params = {}
    f = params.get('filters', [32, 64, 128])
    k = params.get('kernel_size', 7)
    pool = params.get('pool', 2)
    dropout = params.get('dropout', 0.2)

    inp = L.Input(shape=input_shape)
    x = inp
    for _i, fi in enumerate(f):
        x = L.Conv1D(fi, k, padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPool1D(pool)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    # Linear head (no activation) for interpretability
    out = L.Dense(1, use_bias=True)(x)
    return Model(inp, out)

class AddSinusoidalPE(L.Layer):
    """Ajoute un encodage positionnel sin/cos à x (B,T,C). Sortie même shape/dtype que x."""
    def build(self, input_shape):
        self.c = int(input_shape[-1])

    def call(self, x):
        # x: (B,T,C)
        T = tf.shape(x)[1]
        C = tf.shape(x)[-1]
        dtype = x.dtype

        # Positions (T,1)
        pos = tf.cast(tf.range(T)[:, None], dtype)

        # Indices pairs 0,2,4,... pour les fréquences (C//2)
        i = tf.cast(tf.range(0, C, 2)[None, :], dtype)  # (1, ceil(C/2))

        # div_term = exp( -ln(10000) * (2i/C) )
        div = tf.exp(-tf.math.log(tf.constant(10000.0, dtype)) * (i / tf.cast(C, dtype)))  # (1, ceil(C/2))

        sin = tf.sin(pos * div)  # (T, ceil(C/2))
        cos = tf.cos(pos * div)  # (T, ceil(C/2))

        # Interleave sin/cos pour reconstruire (T,C)
        pe_even = sin
        pe_odd  = cos
        pe = tf.reshape(tf.stack([pe_even, pe_odd], axis=-1), (T, -1))  # (T, >=C)
        pe = pe[:, :C]  # tronque si C impair

        pe = tf.expand_dims(pe, 0)  # (1,T,C)
        return x + pe

    def compute_output_shape(self, input_shape):
        return input_shape

def SpectraFormer_block(x, dim, heads, mlp_ratio=4, dw_kernel=7, attn_dropout=0.0, ff_dropout=0.1):
    # Local token mixing before attention (Conformer idea)
    y = L.DepthwiseConv1D(dw_kernel, padding="same")(x)
    y = L.BatchNormalization()(y)
    x = L.Add()([x, y])

    # MHSA
    z = L.LayerNormalization(epsilon=1e-6)(x)
    z = L.MultiHeadAttention(num_heads=heads, key_dim=dim//heads, dropout=attn_dropout)(z, z)
    x = L.Add()([x, z])

    # Position-wise FFN
    z = L.LayerNormalization(epsilon=1e-6)(x)
    z = L.Dense(mlp_ratio*dim, activation="gelu")(z)
    z = L.Dropout(ff_dropout)(z)
    z = L.Dense(dim)(z)
    x = L.Add()([x, z])

    # Local refinement after attention
    z = L.DepthwiseConv1D(dw_kernel, padding="same")(x)
    z = L.BatchNormalization()(z)
    x = L.Add()([x, z])
    return x

@framework('tensorflow')
def spectraformer(input_shape, params: dict = None):
    """
    SpectraFormer for NIRS regression.
    input_shape: (n_wavelengths, 1)
    params:
      dim=128, depth=4, heads=4, patch=4, stem_k=9, dw_kernel=7,
      mlp_ratio=4, attn_dropout=0.0, ff_dropout=0.1, dropout=0.1
    """
    if params is None:
        params = {}
    dim        = params.get("dim", 128)
    depth      = params.get("depth", 4)
    heads      = params.get("heads", 4)
    patch      = params.get("patch", 4)
    stem_k     = params.get("stem_k", 9)
    dw_kernel  = params.get("dw_kernel", 7)
    mlp_ratio  = params.get("mlp_ratio", 4)
    attn_do    = params.get("attn_dropout", 0.0)
    ff_do      = params.get("ff_dropout", 0.1)
    dropout    = params.get("dropout", 0.1)

    inp = L.Input(shape=input_shape)                 # (T, 1)

    # Conv stem (dénoise léger, capte voisinage court)
    x = L.Conv1D(dim//2, stem_k, padding="same", activation="gelu")(inp)
    x = L.BatchNormalization()(x)

    # Patch embedding (patchify + projection)
    x = L.Conv1D(dim, kernel_size=patch, strides=patch, padding="same")(x)  # (T/patch, dim)
    x = L.Activation('gelu')(x)
    x = L.BatchNormalization()(x)
    x = AddSinusoidalPE()(x)
    x = L.Dropout(dropout)(x)

    # Transformer blocks avec mélange local
    for _ in range(depth):
        x = SpectraFormer_block(x, dim=dim, heads=heads, mlp_ratio=mlp_ratio,
                                dw_kernel=dw_kernel, attn_dropout=attn_do, ff_dropout=ff_do)

    # Tête régression
    x = L.LayerNormalization(epsilon=1e-6)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1)(x)
    return Model(inp, out)

######################
## PERPLEXITY MODELS ##
######################
@framework('tensorflow')
def sota_cnn_attention(input_shape, params=None):
    from tensorflow.keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, Input, LayerNormalization, MultiHeadAttention, SpatialDropout1D
    from tensorflow.keras.models import Sequential

    if params is None:
        params = {}
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=32, kernel_size=15, strides=1, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=7, strides=1, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

@framework('tensorflow')
def hybrid_cnn_lstm(input_shape, params=None):
    from tensorflow.keras.layers import LSTM, BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input

    if params is None:
        params = {}
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

##################
## Claude MODELS ##
##################

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    Reshape,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model, Sequential  # noqa: F811

# Assuming @framework decorator exists in your codebase
# from your_framework import framework

@framework('tensorflow')
def resnet1d(input_shape, params=None):
    """
    1D ResNet with skip connections - excellent for deep spectral learning.
    Residual connections help gradient flow and enable deeper architectures.
    """
    if params is None:
        params = {}
    inputs = Input(shape=input_shape)

    # Initial conv
    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)
    x = Conv1D(filters=params.get('filters1', 32), kernel_size=15,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Residual blocks
    n_blocks = params.get('n_residual_blocks', 3)
    filters = [params.get('filters2', 64), params.get('filters3', 128),
               params.get('filters4', 256)]

    for i in range(n_blocks):
        f = filters[i] if i < len(filters) else filters[-1]

        # Residual block
        shortcut = x

        # First conv in block
        x = Conv1D(f, kernel_size=11, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

        # Second conv in block
        x = Conv1D(f, kernel_size=11, padding='same')(x)
        x = BatchNormalization()(x)

        # Match dimensions for skip connection
        if shortcut.shape[-1] != f:
            shortcut = Conv1D(f, kernel_size=1, padding='same')(shortcut)

        # Add skip connection
        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        # Downsampling
        if i < n_blocks - 1:
            x = MaxPooling1D(pool_size=2)(x)

    # Output layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 64), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)
    x = Dense(params.get('dense_units2', 32), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def senet1d(input_shape, params=None):
    """
    1D CNN with Squeeze-and-Excitation blocks for channel attention.
    SE blocks learn to weight spectral features by importance.
    """
    if params is None:
        params = {}
    se_ratio = params.get('se_ratio', 8)

    def se_block(x, filters, ratio=se_ratio):
        """Squeeze-and-Excitation block"""
        se = GlobalAveragePooling1D()(x)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, filters))(se)
        return Multiply()([x, se])

    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # Conv block 1
    x = Conv1D(params.get('filters1', 64), kernel_size=15,
               strides=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = se_block(x, params.get('filters1', 64))
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Conv block 2
    x = Conv1D(params.get('filters2', 128), kernel_size=11,
               strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = se_block(x, params.get('filters2', 128))
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Conv block 3
    x = Conv1D(params.get('filters3', 256), kernel_size=7,
               strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = se_block(x, params.get('filters3', 256))

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 128), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    x = Dense(params.get('dense_units2', 64), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def inception1d(input_shape, params=None):
    """
    Multi-scale 1D Inception network for NIRS.
    Captures features at different spectral resolutions simultaneously.
    """
    if params is None:
        params = {}
    def inception_block(x, filters):
        """Inception module with parallel convolutions"""
        # 1x1 conv
        branch1 = Conv1D(filters // 4, kernel_size=1, padding='same', activation='relu')(x)

        # 1x1 -> 3x3 conv
        branch2 = Conv1D(filters // 4, kernel_size=1, padding='same', activation='relu')(x)
        branch2 = Conv1D(filters // 4, kernel_size=3, padding='same', activation='relu')(branch2)

        # 1x1 -> 5x5 conv
        branch3 = Conv1D(filters // 4, kernel_size=1, padding='same', activation='relu')(x)
        branch3 = Conv1D(filters // 4, kernel_size=5, padding='same', activation='relu')(branch3)

        # 3x3 pool -> 1x1 conv
        branch4 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        branch4 = Conv1D(filters // 4, kernel_size=1, padding='same', activation='relu')(branch4)

        # Concatenate all branches
        output = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        return output

    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # Initial conv
    x = Conv1D(params.get('filters1', 64), kernel_size=15,
               strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Inception blocks
    n_blocks = params.get('n_inception_blocks', 3)
    for i in range(n_blocks):
        filters = params.get(f'inception_filters_{i+1}', 128 * (2**i))
        x = inception_block(x, filters)
        x = BatchNormalization()(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)
        if i < n_blocks - 1:
            x = MaxPooling1D(pool_size=2)(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 128), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    x = Dense(params.get('dense_units2', 64), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def tcn1d(input_shape, params=None):
    """
    Temporal Convolutional Network with dilated causal convolutions.
    Excellent for capturing long-range dependencies in spectra.
    """
    if params is None:
        params = {}
    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # TCN blocks with increasing dilation
    n_blocks = params.get('n_tcn_blocks', 4)
    filters = params.get('filters', 64)
    kernel_size = params.get('kernel_size', 7)

    for i in range(n_blocks):
        dilation_rate = 2 ** i

        # Dilated conv 1
        x = Conv1D(filters, kernel_size, padding='causal',
                   dilation_rate=dilation_rate, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

        # Dilated conv 2
        x = Conv1D(filters, kernel_size, padding='causal',
                   dilation_rate=dilation_rate, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 128), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    x = Dense(params.get('dense_units2', 64), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def attention_cnn1d(input_shape, params=None):
    """
    CNN with multi-head self-attention mechanism.
    Combines local feature extraction (CNN) with global context (attention).
    State-of-the-art for spectral analysis.
    """
    if params is None:
        params = {}
    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # CNN feature extraction
    x = Conv1D(params.get('filters1', 64), kernel_size=15,
               strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    x = Conv1D(params.get('filters2', 128), kernel_size=11,
               strides=2, activation='relu')(x)
    x = BatchNormalization()(x)

    # Multi-head attention
    num_heads = params.get('num_attention_heads', 4)
    key_dim = params.get('attention_key_dim', 32)

    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=params.get('attention_dropout', 0.1)
    )(x, x)

    # Residual connection
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # More CNN layers
    x = Conv1D(params.get('filters3', 256), kernel_size=7,
               strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 128), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    x = Dense(params.get('dense_units2', 64), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def deep_resnet1d(input_shape, params=None):
    """
    Deeper ResNet variant specifically optimized for NIRS.
    More residual blocks with bottleneck architecture.
    Best for complex spectra with subtle patterns.
    """
    if params is None:
        params = {}
    def bottleneck_block(x, filters, downsample=False):
        """Bottleneck residual block"""
        strides = 2 if downsample else 1
        shortcut = x

        # 1x1 conv (compress)
        x = Conv1D(filters // 4, kernel_size=1, strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 3x3 conv
        x = Conv1D(filters // 4, kernel_size=7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 1x1 conv (expand)
        x = Conv1D(filters, kernel_size=1)(x)
        x = BatchNormalization()(x)

        # Match dimensions
        if downsample or shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, kernel_size=1, strides=strides)(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # Initial conv
    x = Conv1D(params.get('filters1', 64), kernel_size=15,
               strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual stages
    filters_list = [params.get('stage1_filters', 64),
                    params.get('stage2_filters', 128),
                    params.get('stage3_filters', 256),
                    params.get('stage4_filters', 512)]

    blocks_per_stage = params.get('blocks_per_stage', 2)

    for stage_idx, filters in enumerate(filters_list):
        for block_idx in range(blocks_per_stage):
            downsample = (block_idx == 0 and stage_idx > 0)
            x = bottleneck_block(x, filters, downsample)
            x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 256), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.4))(x)
    x = Dense(params.get('dense_units2', 128), activation='relu')(x)
    x = Dropout(params.get('dropout_rate', 0.3))(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

##################
## GROK MODELS ##
##################

from tensorflow.keras import backend as K
from tensorflow.keras import layers  # noqa: F811
from tensorflow.keras.layers import Embedding


@framework('tensorflow')
def transformer_nirs(input_shape, params=None):
    """
    Builds a Transformer-based model for NIRS prediction, using encoder blocks with self-attention.
    Suitable for capturing global dependencies in spectral sequences.

    Parameters:
    input_shape (tuple): Shape of the input data, e.g., (num_wavelengths, 1).
    params (dict): Dictionary of parameters for model configuration.

    Returns:
    keras.Model: Compiled model (not compiled here, as per nicon).
    """
    if params is None:
        params = {}
    num_heads = params.get('num_heads', 4)
    ff_dim = params.get('ff_dim', 64)
    num_layers = params.get('num_layers', 2)
    dropout_rate = params.get('dropout_rate', 0.1)
    embed_dim = params.get('embed_dim', 32)  # Projection dimension for attention

    inputs = Input(shape=input_shape)

    # Project input to embed_dim
    x = Conv1D(filters=embed_dim, kernel_size=1, padding='same')(inputs)

    # Add positional encoding
    seq_len = input_shape[0]
    position_embedding = Embedding(input_dim=seq_len, output_dim=embed_dim)
    positions = tf.range(start=0, limit=seq_len, dtype=tf.int32)
    pos_emb = position_embedding(positions)
    x = x + pos_emb  # Broadcast pos_emb to match batch

    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dense(embed_dim)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    x = Flatten()(x)
    x = Dense(params.get('dense_units', 16), activation='sigmoid')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

@framework('tensorflow')
def resnet_nirs(input_shape, params=None):
    """
    Builds a 1D ResNet-style CNN model for NIRS prediction with residual blocks.
    Enables deeper architectures without vanishing gradients.

    Parameters:
    input_shape (tuple): Shape of the input data, e.g., (num_wavelengths, 1).
    params (dict): Dictionary of parameters for model configuration.

    Returns:
    keras.Model: Compiled model (not compiled here, as per nicon).
    """
    if params is None:
        params = {}
    def residual_block(x, filters, kernel_size=3, strides=1):
        shortcut = x
        y = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(filters, kernel_size, padding='same')(y)
        y = BatchNormalization()(y)

        if strides != 1 or K.int_shape(x)[-1] != filters:
            shortcut = Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = Add()([y, shortcut])
        y = Activation('relu')(y)
        return y

    filters1 = params.get('filters1', 32)
    filters2 = params.get('filters2', 64)
    filters3 = params.get('filters3', 128)
    dropout_rate = params.get('dropout_rate', 0.2)
    num_blocks = params.get('num_blocks', 2)  # Per stage

    inputs = Input(shape=input_shape)
    x = Conv1D(filters1, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # Residual stages
    for _ in range(num_blocks):
        x = residual_block(x, filters1)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters2, kernel_size=3, strides=2, padding='same')(x)  # Downsample
    for _ in range(num_blocks):
        x = residual_block(x, filters2)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters3, kernel_size=3, strides=2, padding='same')(x)  # Downsample
    for _ in range(num_blocks):
        x = residual_block(x, filters3)

    x = Flatten()(x)
    x = Dense(params.get('dense_units', 64), activation='sigmoid')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

##################
## DEEPSEEK MODELS ##
##################
@framework('tensorflow')
def nirs_resnet(input_shape, params=None):
    """ResNet-style 1D CNN with residual connections - SOTA for NIRS"""
    if params is None:
        params = {}
    def residual_block(x, filters, kernel_size, dilation_rate=1):
        shortcut = x
        # Main path
        x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

        x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
        x = BatchNormalization()(x)

        # Matching dimensions for shortcut
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, padding='same')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('swish')(x)
        return x

    inputs = Input(shape=input_shape)

    # Initial conv layer
    x = Conv1D(64, 15, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = SpatialDropout1D(0.1)(x)

    # Residual blocks with increasing filters
    filters_list = [64, 128, 256, 512]
    for filters in filters_list:
        x = residual_block(x, filters, 15)
        x = residual_block(x, filters, 11)
        x = residual_block(x, filters, 7)
        # Downsample
        if filters != 512:
            x = MaxPooling1D(2)(x)

    # Attention mechanism
    attention = GlobalAveragePooling1D()(x)
    attention = Dense(x.shape[-1] // 2, activation='swish')(attention)
    attention = Dense(x.shape[-1], activation='sigmoid')(attention)
    x = Multiply()([x, attention])

    # Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

@framework('tensorflow')
def nirs_inception(input_shape, params=None):
    """Inception-style multi-scale feature extraction for NIRS"""
    if params is None:
        params = {}
    def inception_module(x, filters):
        # Branch 1: 1x1 convolution
        branch1 = Conv1D(filters, 1, padding='same', activation='swish')(x)

        # Branch 2: 3x3 convolution
        branch2 = Conv1D(filters, 3, padding='same', activation='swish')(x)

        # Branch 3: 5x5 convolution
        branch3 = Conv1D(filters, 5, padding='same', activation='swish')(x)

        # Branch 4: Max pooling
        branch4 = MaxPooling1D(3, strides=1, padding='same')(x)
        branch4 = Conv1D(filters, 1, padding='same', activation='swish')(branch4)

        # Concatenate all branches
        return Concatenate()([branch1, branch2, branch3, branch4])

    inputs = Input(shape=input_shape)

    # Initial processing
    x = Conv1D(64, 15, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # Inception modules
    x = inception_module(x, 32)
    x = MaxPooling1D(2)(x)
    x = inception_module(x, 64)
    x = MaxPooling1D(2)(x)
    x = inception_module(x, 128)

    # Global context
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])

    # Output layers
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

@framework('tensorflow')
def nirs_transformer_cnn(input_shape, params=None):
    """Transformer + CNN hybrid for capturing both local and global patterns"""

    if params is None:
        params = {}
    def transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = LayerNormalization()(x + attn_output)

        # Feed forward
        ff_output = Dense(ff_dim, activation="swish")(x)
        ff_output = Dense(x.shape[-1])(ff_output)
        x = LayerNormalization()(x + ff_output)
        return x

    inputs = Input(shape=input_shape)

    # CNN feature extraction
    x = Conv1D(64, 15, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 11, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(256, 7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # Positional encoding
    positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
    positions = tf.expand_dims(positions, 0)
    positions = tf.tile(positions, [tf.shape(x)[0], 1])
    position_embedding = Embedding(input_dim=1000, output_dim=256)(positions)
    x = Add()([x, position_embedding])

    # Transformer layers
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=512)
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=512)

    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

@framework('tensorflow')
def nicon_enhanced(input_shape, params=None):
    """Enhanced nicon with attention and better architecture"""
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    # Use Functional API to properly handle attention mechanism
    if params is None:
        params = {}
    inputs = Input(shape=input_shape)

    # Enhanced preprocessing
    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # Multi-scale feature extraction
    x = Conv1D(filters=32, kernel_size=25, strides=3, padding='same', activation="swish")(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=15, strides=2, padding='same', activation="swish")(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Channel attention (SENet-like) - properly implemented in Functional API
    channels = x.shape[-1]
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)

    # Shared MLP for attention
    gap_dense = Dense(channels // 8, activation='swish')(gap)
    gmp_dense = Dense(channels // 8, activation='swish')(gmp)

    gap_output = Dense(channels, activation='sigmoid')(gap_dense)
    gmp_output = Dense(channels, activation='sigmoid')(gmp_dense)

    # Combine and reshape
    attention = Add()([gap_output, gmp_output])
    attention = Reshape((1, channels))(attention)

    # Apply attention
    x = Multiply()([x, attention])

    x = Conv1D(filters=128, kernel_size=7, strides=2, padding='same', activation="swish")(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation="swish")(x)
    x = BatchNormalization()(x)

    # Global context
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 32), activation="swish")(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation="swish")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="NICON_Enhanced")

###################
## GEMINI MODELS ##
####################

import numpy as np  # noqa: F811

# Utility Sub-Functions for 1D SE-ResNet

def se_block_1D(input_tensor, ratio=16):
    """Implements the Squeeze-and-Excitation mechanism for 1D inputs."""
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]

    # Squeeze: Global Average Pooling (GAP)
    se = layers.GlobalAveragePooling1D()(input_tensor)

    # Excitation: Two fully connected layers (Reduce -> Restore dimension)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    # Reshape to (1, filters) for broadcasting multiplication
    se = layers.Reshape((1, filters))(se)

    # Scale: Multiply input tensor by the recalibrated weights
    return layers.Multiply(name="se_scale")([input_tensor, se])

def residual_block_1d(x, filters, kernel_size=17, strides=1, use_se=True, block_name='res'):
    """
    Implements a 1D residual block including convolution, BN, activation, and optional SE.
    """
    x_shortcut = x

    # Main path: Conv -> BN -> Activation -> Conv -> BN
    # Step 1: Convolution
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    x = layers.Activation('relu', name=f'{block_name}_relu1')(x)

    # Step 2: Convolution
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False, name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)

    # Apply SE Block
    if use_se:
        x = se_block_1D(x, ratio=16)

    # Shortcut path processing (for dimension matching)
    if strides!= 1 or x_shortcut.shape[-1]!= filters:
        # 1x1 convolution for dimension matching (projection shortcut) [3]
        x_shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False, name=f'{block_name}_shortcut_conv')(x_shortcut)
        x_shortcut = layers.BatchNormalization(name=f'{block_name}_shortcut_bn')(x_shortcut)

    # Step 3: Add shortcut connection (applied before final ReLU) [4]
    x = layers.Add(name=f'{block_name}_add')([x, x_shortcut])
    x = layers.Activation('relu', name=f'{block_name}_out_relu')(x)

    return x

# Main Model Function
@framework('tensorflow')
def se_resnet(input_shape, params=None):
    """
    Builds the 1D Squeeze-and-Excitation Residual Network (1D SE-ResNet) for NIRS regression.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (wavelengths, 1)).
        params (dict): Dictionary of parameters for model configuration.
            Expected keys: 'num_blocks', 'initial_filters', 'output_dim', 'kernel_size'.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    if params is None:
        params = {}
    num_blocks = params.get('num_blocks', 4)
    initial_filters = params.get('initial_filters', 32)
    kernel_size = params.get('kernel_size', 17)
    output_dim = params.get('output_dim', 1)

    inputs = layers.Input(shape=input_shape)

    # Initial Convolution and Activation (ResNet Front End) [5]
    x = layers.Conv1D(initial_filters, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)

    # Max pooling is often skipped in 1D sequences to preserve resolution,
    # using strides in Conv1D blocks for downsampling instead.

    # Residual Stack
    filters = initial_filters
    for i in range(num_blocks):
        block_name = f'stage{i+1}'

        # Downsampling via stride=2 is applied at the start of a new stage (i>0)
        strides = 2 if i > 0 and i % 2 == 0 else 1
        if strides == 2:
            filters *= 2

        x = residual_block_1d(x, filters, kernel_size=kernel_size, strides=strides, block_name=block_name)

    # Global Pooling and Final Regression Head
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    # Output activation is 'linear' for regression tasks (predicting normalized concentration)
    outputs = layers.Dense(output_dim, activation="linear", name='output_layer')(x)

    return Model(inputs, outputs, name="SE_ResNet_NIRS")

# Utility Sub-Function for 1D Transformer (SpectraTr)

def transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout=0.1, block_name='trans'):
    """
    Implements a single 1D Transformer encoder block.
    """
    # 1. Multi-Head Self-Attention (MHSA)
    norm1 = layers.LayerNormalization(epsilon=1e-6, name=f'{block_name}_norm1')(x)
    attn_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout, name=f'{block_name}_mha'
    )(norm1, norm1)

    # Add & Norm (Residual Connection 1)
    res1 = layers.Add(name=f'{block_name}_add1')([attn_output, x])

    # 2. Feed Forward Network (FFN)
    norm2 = layers.LayerNormalization(epsilon=1e-6, name=f'{block_name}_norm2')(res1)
    # FFN uses 1D convolutions (or Dense layers applied sequence-wise)
    ffn_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", name=f'{block_name}_ffn1')(norm2)
    ffn_output = layers.Dropout(dropout, name=f'{block_name}_ffn_drop')(ffn_output)
    ffn_output = layers.Conv1D(filters=x.shape[-1], kernel_size=1, name=f'{block_name}_ffn2')(ffn_output)

    # Add & Norm (Residual Connection 2)
    return layers.Add(name=f'{block_name}_add2')([ffn_output, res1])

# Main Model Function
@framework('tensorflow')
def spectratr_transformer(input_shape, params=None):
    """
    Builds a 1D Transformer model (SpectraTr adaptation) for NIRS regression.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (wavelengths, 1)).
        params (dict): Dictionary of parameters for model configuration.
            Expected keys: 'patch_size', 'head_size', 'num_heads', 'ff_dim',
                           'num_transformer_blocks', 'mlp_units', 'dropout', 'output_dim'.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    if params is None:
        params = {}
    patch_size = params.get('patch_size', 16)
    head_size = params.get('head_size', 128)
    num_heads = params.get('num_heads', 4)
    ff_dim = params.get('ff_dim', 4)
    num_transformer_blocks = params.get('num_transformer_blocks', 3)
    mlp_units = params.get('mlp_units', )
    dropout = params.get('dropout', 0.1)
    output_dim = params.get('output_dim', 1)

    spectral_length = input_shape[0]
    # Ensure spectral length is divisible by patch_size for simplicity
    if spectral_length % patch_size!= 0:
        raise ValueError(f"Spectral length ({spectral_length}) must be divisible by patch_size ({patch_size}).")

    num_patches = spectral_length // patch_size

    inputs = layers.Input(shape=input_shape)

    # 1. Patch Creation / Embedding (Input to Tokens)
    # Use 1D Conv layer with stride=patch_size to create patches (tokens) and linear projection
    x = layers.Conv1D(
        filters=head_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_projection"
    )(inputs)
    # x shape is now (None, num_patches, head_size)

    # 2. Positional Encoding (Ensuring sequential order is retained)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_emb = layers.Embedding(input_dim=num_patches, output_dim=head_size, name="positional_embedding")(positions)

    # Add Positional Encoding to the patch embeddings
    x = layers.Add(name="patch_pos_add")([x, pos_emb])

    # 3. Transformer Encoder Blocks
    for i in range(num_transformer_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout, block_name=f'trans_block{i+1}')

    # 4. Classification/Regression Head
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Final Multi-Layer Perceptron (MLP) for regression
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu", name=f'dense_head_{dim}')(x)
        x = layers.Dropout(dropout, name=f'dropout_head_{dim}')(x)

    # Output activation is 'linear' for regression tasks
    outputs = layers.Dense(output_dim, activation="linear", name='output_layer')(x)

    return Model(inputs, outputs, name="SpectraTr_Transformer")
