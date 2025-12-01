from typing import Any, Callable, Sequence, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import flax.linen as nn
from nirs4all.utils import framework

# -----------------------------------------------------------------------------
#  Layers & Helpers
# -----------------------------------------------------------------------------

class SpatialDropout1D(nn.Module):
    rate: float = 0.5

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if self.rate == 0.0:
            return x
        # x is (N, L, C). Drop channels -> broadcast over L (dim 1).
        return nn.Dropout(rate=self.rate, broadcast_dims=(1,))(x, deterministic=deterministic)

class DepthwiseConv1D(nn.Module):
    kernel_size: int
    strides: int = 1
    padding: str = 'SAME'
    depth_multiplier: int = 1
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        out_features = in_features * self.depth_multiplier
        x = nn.Conv(
            features=out_features,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding=self.padding,
            feature_group_count=in_features
        )(x)
        if self.activation:
            x = self.activation(x)
        return x

class SeparableConv1D(nn.Module):
    features: int
    kernel_size: int
    strides: int = 1
    padding: str = 'SAME'
    depth_multiplier: int = 1
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        x = DepthwiseConv1D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            depth_multiplier=self.depth_multiplier
        )(x)
        x = nn.Conv(features=self.features, kernel_size=(1,))(x)
        if self.activation:
            x = self.activation(x)
        return x

class GlobalAveragePooling1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis=1)

class MaxPooling1D(nn.Module):
    window_shape: Tuple[int] = (2,)
    strides: Tuple[int] = (2,)
    padding: str = 'SAME'

    @nn.compact
    def __call__(self, x):
        return nn.max_pool(x, window_shape=self.window_shape, strides=self.strides, padding=self.padding)

class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))

class Activation(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x):
        return self.fn(x)

def get_activation(name: str):
    if name is None:
        return None
    name = name.lower()
    if name == "swish":
        return nn.swish
    elif name == "selu":
        return nn.selu
    elif name == "elu":
        return nn.elu
    elif name == "relu":
        return nn.relu
    elif name == "sigmoid":
        return nn.sigmoid
    elif name == "softmax":
        return nn.softmax
    elif name == "tanh":
        return nn.tanh
    else:
        return nn.relu

def get_norm(method: str):
    if method == "LayerNormalization":
        return nn.LayerNorm(epsilon=1e-6)
    return nn.BatchNorm(use_running_average=None)

class DynamicModel(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, train: bool = False):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm):
                x = layer(x, use_running_average=not train)
            elif isinstance(layer, (nn.Dropout, SpatialDropout1D)):
                x = layer(x, deterministic=not train)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, deterministic=not train)
            else:
                x = layer(x)
        return x

class TransformerBlock(nn.Module):
    head_size: int
    num_heads: int
    ff_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Attention
        attn_out = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.head_size * self.num_heads,
            dropout_rate=self.dropout
        )(x, x, deterministic=deterministic)

        x = nn.LayerNorm(epsilon=1e-6)(attn_out)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        res = x + x # Wait, TF code: res = x + inputs. But x here is Dropout(Norm(Attn)).
        # TF code:
        # x = MultiHeadAttention(...)(inputs, inputs)
        # x = LayerNormalization(...)(x)
        # x = Dropout(...)(x)
        # res = x + inputs

        # So I need to keep 'inputs' (which is 'x' at start of function)
        # But I overwrote 'x'.
        # Let's fix.
        pass # Placeholder, will implement in _build_transformer logic or separate class

# Re-implement TransformerBlock properly
class TransformerBlockImpl(nn.Module):
    head_size: int
    num_heads: int
    ff_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        inputs = x
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.head_size * self.num_heads,
            dropout_rate=self.dropout
        )(inputs, inputs, deterministic=deterministic)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        res = x + inputs

        # Feed Forward
        x = nn.Conv(features=self.ff_dim, kernel_size=(1,), activation=nn.relu)(res)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Conv(features=inputs.shape[-1], kernel_size=(1,))(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        return x + res

# -----------------------------------------------------------------------------
#  Model Builders
# -----------------------------------------------------------------------------

def _build_decon(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.2)))

    # Block 1
    layers.append(DepthwiseConv1D(kernel_size=7, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(DepthwiseConv1D(kernel_size=7, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(MaxPooling1D(window_shape=(2,), strides=(2,), padding='SAME'))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Block 2
    layers.append(DepthwiseConv1D(kernel_size=5, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(DepthwiseConv1D(kernel_size=5, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(MaxPooling1D(window_shape=(2,), strides=(2,), padding='SAME'))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Block 3
    layers.append(DepthwiseConv1D(kernel_size=9, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(DepthwiseConv1D(kernel_size=9, padding='SAME', depth_multiplier=2, activation=nn.relu))
    layers.append(MaxPooling1D(window_shape=(2,), strides=(2,), padding='SAME'))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Separable + Conv
    layers.append(SeparableConv1D(features=64, kernel_size=3, padding='SAME', depth_multiplier=1, activation=nn.relu))
    layers.append(nn.Conv(features=32, kernel_size=(3,), padding='SAME'))
    layers.append(Activation(nn.relu))
    layers.append(MaxPooling1D(window_shape=(5,), strides=(3,), padding='SAME'))

    layers.append(SpatialDropout1D(rate=0.1))
    layers.append(Flatten())

    layers.append(nn.Dense(features=128))
    layers.append(Activation(nn.relu))
    layers.append(nn.Dense(features=32))
    layers.append(Activation(nn.relu))
    layers.append(nn.Dropout(rate=0.2))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_decon_sep(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.2)))

    # Sep 1
    f1 = params.get('filters1', 64)
    k1 = params.get('kernel_size1', 3)
    s1 = params.get('strides1', 2)
    d1 = params.get('depth_multiplier1', 32)
    layers.append(SeparableConv1D(features=f1, kernel_size=k1, strides=s1, padding='SAME', depth_multiplier=d1, activation=nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Sep 2
    f2 = params.get('filters2', 64)
    k2 = params.get('kernel_size2', 3)
    s2 = params.get('strides2', 2)
    d2 = params.get('depth_multiplier2', 32)
    layers.append(SeparableConv1D(features=f2, kernel_size=k2, strides=s2, padding='SAME', depth_multiplier=d2, activation=nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Sep 3
    f3 = params.get('filters3', 64)
    k3 = params.get('kernel_size3', 3)
    d3 = params.get('depth_multiplier3', 32)
    layers.append(SeparableConv1D(features=f3, kernel_size=k3, strides=1, padding='SAME', depth_multiplier=d3, activation=nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Sep 4
    f4 = params.get('filters4', 64)
    k4 = params.get('kernel_size4', 3)
    d4 = params.get('depth_multiplier4', 32)
    layers.append(SeparableConv1D(features=f4, kernel_size=k4, strides=1, padding='SAME', depth_multiplier=d4, activation=nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    # Conv 5
    f5 = params.get('filters5', 32)
    k5 = params.get('kernel_size5', 5)
    s5 = params.get('strides5', 2)
    layers.append(nn.Conv(features=f5, kernel_size=(k5,), strides=(s5,), padding='SAME'))
    layers.append(Activation(nn.relu))

    layers.append(Flatten())
    layers.append(nn.BatchNorm(use_running_average=None))

    du = params.get('dense_units', 32)
    layers.append(nn.Dense(features=du))
    layers.append(Activation(nn.relu))
    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_nicon(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    layers.append(nn.Conv(features=f1, kernel_size=(15,), strides=(5,)))
    layers.append(Activation(nn.selu))

    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv(features=f2, kernel_size=(21,), strides=(3,)))
    layers.append(Activation(nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    f3 = params.get('filters3', 32)
    layers.append(nn.Conv(features=f3, kernel_size=(5,), strides=(3,)))
    layers.append(Activation(nn.elu))
    layers.append(nn.BatchNorm(use_running_average=None))

    layers.append(Flatten())

    du = params.get('dense_units', 16)
    layers.append(nn.Dense(features=du))
    layers.append(Activation(nn.sigmoid))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_customizable_nicon(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    k1 = params.get('kernel_size1', 15)
    s1 = params.get('strides1', 5)
    act1 = get_activation(params.get('activation1', "selu"))
    layers.append(nn.Conv(features=f1, kernel_size=(k1,), strides=(s1,)))
    if act1: layers.append(Activation(act1))

    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    k2 = params.get('kernel_size2', 21)
    s2 = params.get('strides2', 3)
    act2 = get_activation(params.get('activation2', "relu"))
    layers.append(nn.Conv(features=f2, kernel_size=(k2,), strides=(s2,)))
    if act2: layers.append(Activation(act2))
    norm1 = get_norm(params.get('normalization_method1', "BatchNormalization"))
    layers.append(norm1)

    f3 = params.get('filters3', 32)
    k3 = params.get('kernel_size3', 5)
    s3 = params.get('strides3', 3)
    act3 = get_activation(params.get('activation3', "elu"))
    layers.append(nn.Conv(features=f3, kernel_size=(k3,), strides=(s3,)))
    if act3: layers.append(Activation(act3))
    norm2 = get_norm(params.get('normalization_method2', "BatchNormalization"))
    layers.append(norm2)

    layers.append(Flatten())

    du = params.get('dense_units', 16)
    act_d = get_activation(params.get('dense_activation', "sigmoid"))
    layers.append(nn.Dense(features=du))
    if act_d: layers.append(Activation(act_d))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_thin_nicon(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.08)))

    f1 = params.get('filters1', 8)
    layers.append(nn.Conv(features=f1, kernel_size=(7,), strides=(3,)))
    layers.append(Activation(nn.selu))

    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv(features=f2, kernel_size=(11,), strides=(2,)))
    layers.append(Activation(nn.relu))
    layers.append(nn.BatchNorm(use_running_average=None))

    f3 = params.get('filters3', 32)
    layers.append(nn.Conv(features=f3, kernel_size=(3,), strides=(2,)))
    layers.append(Activation(nn.elu))
    layers.append(nn.BatchNorm(use_running_average=None))

    layers.append(Flatten())

    du = params.get('dense_units', 16)
    layers.append(nn.Dense(features=du))
    layers.append(Activation(nn.sigmoid))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_nicon_vg(input_shape, params, num_classes=1):
    layers = []
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.2)))

    f1 = params.get('filters1', 64)
    layers.append(nn.Conv(features=f1, kernel_size=(3,), padding='SAME'))
    layers.append(Activation(nn.swish))

    f2 = params.get('filters2', 64)
    layers.append(nn.Conv(features=f2, kernel_size=(3,), padding='SAME'))
    layers.append(Activation(nn.swish))

    layers.append(MaxPooling1D(window_shape=(5,), strides=(3,)))

    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.2)))

    f3 = params.get('filters3', 128)
    layers.append(nn.Conv(features=f3, kernel_size=(3,), padding='SAME'))
    layers.append(Activation(nn.swish))

    f4 = params.get('filters4', 128)
    layers.append(nn.Conv(features=f4, kernel_size=(3,), padding='SAME'))
    layers.append(Activation(nn.swish))

    layers.append(MaxPooling1D(window_shape=(5,), strides=(3,)))

    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout', 0.2)))
    layers.append(Flatten())

    du1 = params.get('dense_units1', 1024)
    layers.append(nn.Dense(features=du1))
    layers.append(Activation(nn.relu))

    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    du2 = params.get('dense_units2', 1024)
    layers.append(nn.Dense(features=du2))
    layers.append(Activation(nn.relu))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_customizable_decon(input_shape, params, num_classes=1):
    layers = []

    # Block 1
    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout1', 0.2)))

    k1 = params.get('kernel_size1', 7)
    d1 = params.get('depth_multiplier1', 2)
    act1 = get_activation(params.get('activationDCNN1', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k1, padding='SAME', depth_multiplier=d1, activation=act1))

    k2 = params.get('kernel_size2', 7)
    d2 = params.get('depth_multiplier2', 2)
    act2 = get_activation(params.get('activationDCNN2', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k2, padding='SAME', depth_multiplier=d2, activation=act2))

    p1 = params.get('pool_size1', 2)
    s1 = params.get('strides1', 2)
    layers.append(MaxPooling1D(window_shape=(p1,), strides=(s1,)))

    layers.append(nn.LayerNorm(epsilon=1e-6))

    # Block 2
    k3 = params.get('kernel_size3', 5)
    d3 = params.get('depth_multiplier3', 2)
    act3 = get_activation(params.get('activationDCNN3', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k3, padding='SAME', depth_multiplier=d3, activation=act3))

    k4 = params.get('kernel_size4', 5)
    d4 = params.get('depth_multiplier4', 2)
    act4 = get_activation(params.get('activationDCNN4', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k4, padding='SAME', depth_multiplier=d4, activation=act4))

    p2 = params.get('pool_size2', 2)
    s2 = params.get('strides2', 2)
    layers.append(MaxPooling1D(window_shape=(p2,), strides=(s2,)))

    layers.append(nn.LayerNorm(epsilon=1e-6))

    # Block 3
    k5 = params.get('kernel_size5', 9)
    d5 = params.get('depth_multiplier5', 2)
    act5 = get_activation(params.get('activationDCNN5', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k5, padding='SAME', depth_multiplier=d5, activation=act5))

    k6 = params.get('kernel_size6', 9)
    d6 = params.get('depth_multiplier6', 2)
    act6 = get_activation(params.get('activationDCNN6', "relu"))
    layers.append(DepthwiseConv1D(kernel_size=k6, padding='SAME', depth_multiplier=d6, activation=act6))

    p3 = params.get('pool_size3', 2)
    s3 = params.get('strides3', 2)
    layers.append(MaxPooling1D(window_shape=(p3,), strides=(s3,)))

    layers.append(nn.LayerNorm(epsilon=1e-6))

    # Final Conv
    sep_f = params.get('separable_filters', 64)
    sep_k = params.get('separable_kernel_size', 3)
    sep_d = params.get('separable_depth_multiplier', 1)
    act_cnn1 = get_activation(params.get('activationCNN1', "relu"))
    layers.append(SeparableConv1D(features=sep_f, kernel_size=sep_k, padding='SAME', depth_multiplier=sep_d, activation=act_cnn1))

    conv_f = params.get('conv_filters', 32)
    conv_k = params.get('conv_kernel_size', 3)
    layers.append(nn.Conv(features=conv_f, kernel_size=(conv_k,), padding='SAME'))

    fp = params.get('final_pool_size', 5)
    fs = params.get('final_pool_strides', 3)
    layers.append(MaxPooling1D(window_shape=(fp,), strides=(fs,)))

    layers.append(SpatialDropout1D(rate=params.get('spatial_dropout2', 0.1)))
    layers.append(Flatten())

    du1 = params.get('dense_units1', 128)
    act_d1 = get_activation(params.get('activationDense1', "relu"))
    layers.append(nn.Dense(features=du1))
    if act_d1: layers.append(Activation(act_d1))

    du2 = params.get('dense_units2', 32)
    act_d2 = get_activation(params.get('activationDense2', "relu"))
    layers.append(nn.Dense(features=du2))
    if act_d2: layers.append(Activation(act_d2))

    layers.append(nn.Dropout(rate=params.get('dropout_rate', 0.2)))

    out_units = params.get('output_units', 1)
    act_out = get_activation(params.get('activationDense3', "sigmoid"))

    if num_classes == 1:
        layers.append(nn.Dense(features=out_units))
        if act_out: layers.append(Activation(act_out))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

def _build_transformer(input_shape, params, num_classes=1):
    layers = []

    head_size = params.get("head_size", 16)
    num_heads = params.get("num_heads", 2)
    ff_dim = params.get("ff_dim", 8)
    num_blocks = params.get("num_transformer_blocks", 1)
    dropout = params.get("dropout", 0.05)

    d_model = head_size * num_heads

    # Project input
    layers.append(nn.Conv(features=d_model, kernel_size=(1,)))

    # Transformer blocks
    for _ in range(num_blocks):
        layers.append(TransformerBlockImpl(
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        ))

    layers.append(GlobalAveragePooling1D())

    mlp_units = params.get("mlp_units", [32, 8])
    mlp_dropout = params.get("mlp_dropout", 0.1)

    for dim in mlp_units:
        layers.append(nn.Dense(features=dim))
        layers.append(Activation(nn.relu))
        layers.append(nn.Dropout(rate=mlp_dropout))

    if num_classes == 1:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    elif num_classes == 2:
        layers.append(nn.Dense(features=1))
        layers.append(Activation(nn.sigmoid))
    else:
        layers.append(nn.Dense(features=num_classes))
        layers.append(Activation(nn.softmax))

    return DynamicModel(layers=layers)

# -----------------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------------

@framework("jax")
def decon(input_shape, params={}):
    return _build_decon(input_shape, params, num_classes=1)

@framework("jax")
def decon_classification(input_shape, num_classes=2, params={}):
    return _build_decon(input_shape, params, num_classes=num_classes)

@framework("jax")
def decon_Sep(input_shape, params={}):
    return _build_decon_sep(input_shape, params, num_classes=1)

@framework("jax")
def decon_Sep_classification(input_shape, num_classes=2, params={}):
    return _build_decon_sep(input_shape, params, num_classes=num_classes)

@framework("jax")
def nicon(input_shape, params={}):
    return _build_nicon(input_shape, params, num_classes=1)

@framework("jax")
def nicon_classification(input_shape, num_classes=2, params={}):
    return _build_nicon(input_shape, params, num_classes=num_classes)

@framework("jax")
def customizable_nicon(input_shape, params={}):
    return _build_customizable_nicon(input_shape, params, num_classes=1)

@framework("jax")
def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_nicon(input_shape, params, num_classes=num_classes)

@framework("jax")
def thin_nicon(input_shape, params={}):
    return _build_thin_nicon(input_shape, params, num_classes=1)

@framework("jax")
def nicon_VG(input_shape, params={}):
    return _build_nicon_vg(input_shape, params, num_classes=1)

@framework("jax")
def nicon_VG_classification(input_shape, num_classes=2, params={}):
    return _build_nicon_vg(input_shape, params, num_classes=num_classes)

@framework("jax")
def customizable_decon(input_shape, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=1)

@framework("jax")
def customizable_decon_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=num_classes)

@framework("jax")
def decon_layer_classification(input_shape, num_classes=2, params={}):
    return _build_customizable_decon(input_shape, params, num_classes=num_classes)

@framework("jax")
def transformer(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("jax")
def transformer_VG(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("jax")
def transformer_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)

@framework("jax")
def transformer_VG_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)

@framework("jax")
def transformer_model(input_shape, params={}):
    return _build_transformer(input_shape, params, num_classes=1)

@framework("jax")
def transformer_model_classification(input_shape, num_classes=2, params={}):
    return _build_transformer(input_shape, params, num_classes=num_classes)
