from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Add, Multiply, Concatenate, Reshape, Average, Lambda,
    AveragePooling1D, ZeroPadding1D, Activation, AlphaDropout, Flatten
)
import tensorflow as tf
from keras.initializers import lecun_normal, he_normal
from nirs4all.utils.backend_utils import framework


@framework('tensorflow')
def nicon_auto_norm(input_shape, params={}):
    model = Sequential([
        Input(shape=input_shape),
        SpatialDropout1D(params.get('spatial_dropout', 0.08)),

        Conv1D(params.get('filters1', 8), 15, strides=5,
            activation="selu", kernel_initializer=lecun_normal()),
        AlphaDropout(params.get('dropout_rate', 0.2)),

        Conv1D(params.get('filters2', 64), 21, strides=3,
            activation="selu", kernel_initializer=lecun_normal()),
        Conv1D(params.get('filters3', 32), 5, strides=3,
            activation="selu", kernel_initializer=lecun_normal()),

        Flatten(),
        Dense(params.get('dense_units', 16), activation="selu",
            kernel_initializer=lecun_normal()),
        Dense(1, activation="sigmoid")  # ou "linear" si Y non borné
    ])
    return model


@framework('tensorflow')
def nicon_batch_norm(input_shape, params={}):
    model = Sequential([
        Input(shape=input_shape),
        SpatialDropout1D(params.get('spatial_dropout', 0.08)),

        Conv1D(params.get('filters1', 8), 15, strides=5,
            activation="relu", kernel_initializer=he_normal()),
        Dropout(params.get('dropout_rate', 0.2)),

        Conv1D(params.get('filters2', 64), 21, strides=3,
            activation="relu", kernel_initializer=he_normal()),
        BatchNormalization(),

        Conv1D(params.get('filters3', 32), 5, strides=3,
            activation="elu", kernel_initializer=he_normal()),
        BatchNormalization(),

        Flatten(),
        Dense(params.get('dense_units', 16), activation="relu",
            kernel_initializer=he_normal()),
        Dense(1, activation="sigmoid")
    ])
    return model


@framework('tensorflow')
def nicon_improved(input_shape, params={}):
    """
    NIRS-optimized CNN with spectral-aware features:
    - Residual connections to preserve baseline
    - Multi-scale feature extraction (sharp + broad peaks)
    - Improved channel attention
    - Derivative-aware branches
    - Minimal aggressive downsampling
    """
    inputs = Input(shape=input_shape)

    # Wavelength positional encoding
    def add_wavelength_encoding(x):
        """Add sinusoidal positional encoding for wavelength information"""
        length = tf.shape(x)[1]
        positions = tf.range(0, length, dtype=tf.float32)
        # Normalize positions
        positions = positions / tf.cast(length, tf.float32)
        # Multiple frequencies
        encoding = tf.sin(positions * 3.14159) * 0.1
        encoding = tf.expand_dims(tf.expand_dims(encoding, 0), -1)
        return x + encoding

    # Add positional information
    x = Lambda(add_wavelength_encoding)(inputs)
    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(x)

    # === MULTI-SCALE INCEPTION BLOCK 1 ===
    # Capture peaks at different widths (narrow, medium, broad)
    branch1 = Conv1D(16, kernel_size=5, padding='same', activation='swish')(x)
    branch2 = Conv1D(16, kernel_size=11, padding='same', activation='swish')(x)
    branch3 = Conv1D(16, kernel_size=21, padding='same', activation='swish')(x)

    # Derivative branch (spectral change detection)
    diff = Lambda(lambda z: tf.concat([z[:, 1:, :] - z[:, :-1, :],
                                       tf.zeros_like(z[:, :1, :])], axis=1))(x)
    branch4 = Conv1D(16, kernel_size=11, padding='same', activation='swish')(diff)

    x = Concatenate()([branch1, branch2, branch3, branch4])  # 64 filters
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.15))(x)

    # === RESIDUAL BLOCK 1 ===
    shortcut1 = x
    x = Conv1D(64, kernel_size=15, strides=1, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=15, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut1])  # Residual connection preserves baseline
    x = Activation('swish')(x)

    # Gentle downsampling (not stride!)
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(params.get('dropout_rate', 0.15))(x)

    # === RESIDUAL BLOCK 2 with Channel Attention ===
    shortcut2 = Conv1D(128, kernel_size=1)(x)  # Match dimensions

    x = Conv1D(128, kernel_size=11, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=11, padding='same')(x)
    x = BatchNormalization()(x)

    # Multi-scale Channel Attention (better than your version)
    channels = 128
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)

    # Also consider local pooling at different scales
    pool2 = AveragePooling1D(pool_size=2)(x)
    gap2 = GlobalAveragePooling1D()(pool2)

    # Combine multi-scale context
    combined = Concatenate()([gap, gmp, gap2])

    # Shared MLP for attention
    attention = Dense(channels // 8, activation='swish')(combined)
    attention = Dense(channels, activation='sigmoid')(attention)
    attention = Reshape((1, channels))(attention)

    # Apply attention
    x = Multiply()([x, attention])
    x = Add()([x, shortcut2])  # Residual connection
    x = Activation('swish')(x)

    # Another gentle downsampling
    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # === RESIDUAL BLOCK 3 ===
    shortcut3 = Conv1D(256, kernel_size=1)(x)

    x = Conv1D(256, kernel_size=7, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut3])
    x = Activation('swish')(x)

    # === OUTPUT HEAD ===
    # Use both average and max pooling for global context
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])

    # Dense layers with decreasing regularization
    x = Dense(params.get('dense_units', 128), activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='swish')(x)

    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs, name="NICON_Improved")


@framework('tensorflow')
def nicon_lightweight(input_shape, params={}):
    """
    Lighter version with key improvements but faster training.
    Good for quick iterations or smaller datasets.
    """
    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # === MULTI-SCALE BLOCK ===
    branch1 = Conv1D(16, kernel_size=7, padding='same', activation='swish')(x)
    branch2 = Conv1D(16, kernel_size=15, padding='same', activation='swish')(x)
    x = Concatenate()([branch1, branch2])  # 32 filters
    x = BatchNormalization()(x)

    # === RESIDUAL BLOCK with SE Attention ===
    shortcut = Conv1D(64, kernel_size=1, strides=2)(x)

    x = Conv1D(64, kernel_size=11, strides=2, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Squeeze-and-Excitation (corrected version)
    channels = 64
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    combined = Average()([gap, gmp])  # Average instead of Add

    attention = Dense(channels // 8, activation='swish')(combined)
    attention = Dense(channels, activation='sigmoid')(attention)
    attention = Reshape((1, channels))(attention)

    x = Multiply()([x, attention])
    x = Add()([x, shortcut])  # Residual
    x = Activation('swish')(x)

    # === ANOTHER RESIDUAL BLOCK ===
    shortcut2 = Conv1D(128, kernel_size=1, strides=2)(x)

    x = Conv1D(128, kernel_size=7, strides=2, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=7, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut2])
    x = Activation('swish')(x)

    # === OUTPUT ===
    x = GlobalAveragePooling1D()(x)
    x = Dense(params.get('dense_units', 64), activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='swish')(x)

    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs, name="NICON_Lightweight")


@framework('tensorflow')
def nicon_experimental(input_shape, params={}):
    """
    Experimental architecture with:
    - Self-attention for global spectral context
    - Explicit derivative learning
    - Adaptive receptive fields
    """
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(params.get('spatial_dropout', 0.1))(inputs)

    # === DERIVATIVE-AWARE BRANCHES ===
    # Raw spectrum
    raw = Conv1D(32, kernel_size=15, padding='same', activation='swish')(x)

    # First derivative approximation
    diff1 = Lambda(lambda z: tf.concat([z[:, 1:, :] - z[:, :-1, :],
                                        tf.zeros_like(z[:, :1, :])], axis=1))(x)
    deriv1 = Conv1D(32, kernel_size=11, padding='same', activation='swish')(diff1)

    # Second derivative approximation
    diff2 = Lambda(lambda z: tf.concat([z[:, 2:, :] - 2*z[:, 1:-1, :] + z[:, :-2, :],
                                        tf.zeros_like(z[:, :2, :])], axis=1))(x)
    deriv2 = Conv1D(32, kernel_size=7, padding='same', activation='swish')(diff2)

    x = Concatenate()([raw, deriv1, deriv2])  # 96 filters
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.15))(x)

    # === SELF-ATTENTION for global context ===
    # Conv to reduce dimensionality first
    x = Conv1D(64, kernel_size=11, strides=2, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)

    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=params.get('num_heads', 4),
        key_dim=params.get('key_dim', 16),
        dropout=0.1
    )(x, x)

    x = Add()([x, attn_output])  # Residual
    x = LayerNormalization()(x)

    # === MORE CONVOLUTIONS ===
    x = Conv1D(128, kernel_size=7, strides=2, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    x = Conv1D(256, kernel_size=5, strides=2, padding='same', activation='swish')(x)
    x = BatchNormalization()(x)

    # === OUTPUT ===
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])

    x = Dense(params.get('dense_units', 128), activation='swish')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs, name="NICON_Experimental")


# ============================================================
# USAGE RECOMMENDATIONS
# ============================================================

"""
RECOMMENDED ORDER TO TRY:

1. **nicon_improved** - Best balance of performance and spectral-awareness
   - Multi-scale inception blocks
   - Proper residual connections
   - Improved multi-scale channel attention
   - Derivative-aware branch
   - Minimal aggressive downsampling

2. **nicon_lightweight** - For quick iterations or small datasets
   - Faster training
   - Core improvements retained
   - Good for initial experiments

3. **nicon_experimental** - When you need maximum performance
   - Self-attention for global context
   - Explicit derivative branches (1st + 2nd)
   - Best for complex spectra
   - Slower but more powerful

KEY DIFFERENCES FROM YOUR nicon_enhanced:
✓ Residual connections preserve baseline information
✓ Multi-scale convolutions adapt to different peak widths
✓ Proper attention averaging (not adding)
✓ Derivative-aware branches (chemically meaningful)
✓ Gentler downsampling (pooling not strides)
✓ Multi-scale attention (considers local + global context)
✓ Wavelength positional encoding

TRAINING TIPS FOR NIRS:
- Use learning rate warmup (helps with noisy spectra)
- Consider focal loss if you have hard examples
- Monitor validation on different spectral regions separately
- Use ensemble of these models + your preprocessing pipeline
"""