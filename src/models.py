"""
Model architecture definitions.
Contains Sequential CNN and Custom ResNet implementations.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    CLR_CONFIG,
    SGD_CONFIG,
    get_image_shape
)


class CyclicLR(Callback):
    """
    Cyclic Learning Rate callback.
    Implements triangular cyclic learning rate policy.
    """
    
    def __init__(
        self, 
        base_lr: float = 0.001, 
        max_lr: float = 0.006, 
        step_size: float = 2000.,
        mode: str = 'triangular',
        gamma: float = 1.,
        scale_fn=None,
        scale_mode: str = 'cycle'
    ):
        """
        Initialize Cyclic LR callback.
        
        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Number of iterations in half cycle
            mode: One of 'triangular', 'triangular2', 'exp_range'
            gamma: Constant for exp_range mode
            scale_fn: Custom scaling function
            scale_mode: 'cycle' or 'iterations'
        """
        super(CyclicLR, self).__init__()
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
            
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        
    def clr(self):
        """Calculate current learning rate."""
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
        
        return lr
    
    def on_train_begin(self, logs=None):
        """Set initial learning rate."""
        logs = logs or {}
        
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
    
    def on_batch_end(self, epoch, logs=None):
        """Update learning rate after each batch."""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


def build_sequential_model() -> Model:
    """
    Build a sequential CNN model.
    Architecture: Multiple Conv-BN blocks with MaxPooling and Dropout.
    
    Returns:
        Compiled Keras model
    """
    config = MODEL_CONFIG['sequential']
    input_shape = get_image_shape()
    num_classes = DATASET_CONFIG['num_classes']
    dropout_rate = config['dropout_rate']
    
    model = Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='norm1'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.BatchNormalization(name='norm2'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        layers.BatchNormalization(name='norm3'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', name='conv4'),
        layers.BatchNormalization(name='norm4'),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv5'),
        layers.BatchNormalization(name='norm5'),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv6'),
        layers.BatchNormalization(name='norm6'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', name='conv7'),
        layers.BatchNormalization(name='norm7'),
        layers.Conv2D(256, (3, 3), activation='relu', name='conv8'),
        layers.BatchNormalization(name='norm8'),
        layers.Conv2D(256, (3, 3), activation='relu', name='conv9'),
        layers.BatchNormalization(name='norm9'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        
        # Output block
        layers.Conv2D(num_classes, (1, 1), activation='relu', name='conv10'),
        layers.BatchNormalization(name='norm10'),
        layers.Conv2D(num_classes, kernel_size=(6, 9), name='conv11'),
        layers.Flatten(),
        layers.Activation('softmax')
    ], name='sequential_cnn')
    
    return model


def conv_bn_relu(input_tensor, filters: int, block_no: int):
    """
    Convolutional block with batch normalization and ReLU.
    
    Args:
        input_tensor: Input tensor
        filters: Number of filters
        block_no: Block number for naming
        
    Returns:
        Output tensor
    """
    x = layers.Conv2D(
        filters, (3, 3),
        kernel_initializer='glorot_uniform',
        padding='same',
        name=f'begin_block{block_no}_conv1',
        use_bias=False
    )(input_tensor)
    x = layers.BatchNormalization(name=f'begin_block{block_no}_norm1')(x)
    x = layers.ReLU()(x)
    return x


def add_resblock(
    input_tensor,
    dropout_rate=None,
    num_layers: int = 2,
    block_no: int = 1,
    l2_reg: float = 0.001,
    first_block: bool = False,
    final_block: bool = False
):
    """
    Add a residual block with skip connections.
    
    Args:
        input_tensor: Input tensor
        dropout_rate: Dropout rate (None for no dropout)
        num_layers: Number of layers in the block
        block_no: Block number for naming
        l2_reg: L2 regularization factor
        first_block: Whether this is the first block
        final_block: Whether this is the final block
        
    Returns:
        Output tensor
    """
    channels = int(input_tensor.shape[3])
    ch_out = channels * 2
    
    temp = input_tensor
    
    for layer_idx in range(num_layers):
        filters = int(ch_out * (2 ** layer_idx))
        
        x = layers.Conv2D(
            filters, (3, 3),
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,
            padding='same',
            name=f'res_block{block_no}_conv{layer_idx + 1}'
        )(temp)
        
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x)
        
        x = layers.BatchNormalization(
            name=f'res_block{block_no}_BN{layer_idx + 1}'
        )(x)
        x = layers.ReLU(name=f'res_block{block_no}_relu{layer_idx + 1}')(x)
        
        temp = x
    
    # Concatenate skip connection
    concat = layers.Concatenate(axis=-1)([input_tensor, temp])
    
    if not final_block:
        # Transition layer
        tr_layer = layers.Conv2D(
            ch_out, (1, 1),
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,
            padding='same',
            name=f'res_block{block_no}_transition'
        )(concat)
        tr_layer = layers.BatchNormalization(
            name=f'res_block_transition1x1{block_no}_BN'
        )(tr_layer)
        tr_layer = layers.ReLU(
            name=f'res_block_transition1x1{block_no}_relu'
        )(tr_layer)
        
        return layers.MaxPooling2D(pool_size=(2, 2))(tr_layer)
    else:
        return concat


def build_resnet_model() -> Model:
    """
    Build a custom ResNet model.
    Architecture inspired by ResNet18 with residual blocks.
    
    Returns:
        Compiled Keras model
    """
    config = MODEL_CONFIG['resnet']
    num_classes = DATASET_CONFIG['num_classes']
    l2_reg = config['l2_regularization']
    dropout = config['dropout_rate']
    
    # Input layer with flexible shape
    input_layer = layers.Input(shape=(None, None, 3))
    
    # Initial convolution
    x = conv_bn_relu(input_layer, filters=32, block_no=1)
    
    # Residual blocks
    x = add_resblock(x, dropout_rate=dropout, num_layers=2, 
                     block_no=1, l2_reg=l2_reg, first_block=True)
    x = add_resblock(x, dropout_rate=dropout, num_layers=2, 
                     block_no=2, l2_reg=l2_reg)
    x = add_resblock(x, dropout_rate=dropout, num_layers=2, 
                     block_no=3, l2_reg=l2_reg)
    x = add_resblock(x, dropout_rate=dropout, num_layers=2, 
                     block_no=4, l2_reg=l2_reg, final_block=True)
    
    # Output layers
    x = layers.Conv2D(num_classes, (1, 1), name='ch_size_no_classes', use_bias=False)(x)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Softmax()(x)
    
    model = Model(inputs=[input_layer], outputs=[output_layer], name='custom_resnet')
    
    return model


def compile_model(model: Model, model_type: str = 'sequential') -> Model:
    """
    Compile model with appropriate optimizer and loss.
    
    Args:
        model: Keras model to compile
        model_type: Type of model ('sequential' or 'resnet')
        
    Returns:
        Compiled model
    """
    if model_type == 'sequential':
        optimizer = Adam(learning_rate=0.001)
    else:  # resnet
        optimizer = SGD(
            learning_rate=SGD_CONFIG['learning_rate'],
            momentum=SGD_CONFIG['momentum'],
            nesterov=SGD_CONFIG['nesterov']
        )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_type: str = 'sequential') -> Model:
    """
    Get a compiled model of specified type.
    
    Args:
        model_type: Type of model ('sequential' or 'resnet')
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'sequential':
        model = build_sequential_model()
    elif model_type == 'resnet':
        model = build_resnet_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model, model_type)
    
    print(f"\n{model_type.upper()} Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

