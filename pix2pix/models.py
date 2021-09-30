from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.initializers import RandomNormal

def block_conv2D(filters, kernel_size=4, strides=(2,2), with_batch_norm=True, relu_alpha=0.3):
    '''
        Return a block of layers consisting of Conv2D-BatchNormal-LeakyReLU or Conv2D-LeakyReLU layers
    '''
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    block = Sequential()
    block.add(layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))
    if with_batch_norm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(alpha=relu_alpha))
    
    return block


def block_conv2D_transpose(filters, kernel_size=4, strides=(2,2), with_dropout=False, dropout_rate=0.5):
    '''
        Return a block of layers consisting of Conv2D-BatchNorm-Dropout-ReLU or Conv2D-BatchNorme-ReLU
    '''
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    block = Sequential()
    block.add(layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))
    block.add(layers.BatchNormalization())
    if with_dropout:
        block.add(layers.Dropout(dropout_rate))
    block.add(layers.ReLU())

    return block


def make_generator_encoder_model(strides=(2,2)):
    inputs = layers.Input(shape=(256, 256, 3))
    # Entry block
    x = block_conv2D(64, with_batch_norm=False)(inputs)
    # Shape = (128, 128, 64)
    x = block_conv2D(128)(x)
    # Shape = (64, 64, 128)
    x = block_conv2D(256)(x)
    # Shape = (32, 32, 256)
    x = block_conv2D(512)(x)
    # Shape = (16, 16, 512)
    x = block_conv2D(512)(x)
    # Shape = (8, 8, 512)
    x = block_conv2D(512)(x)
    # Shape = (4, 4, 512)
    x = block_conv2D(512)(x)
    # Shape = (2, 2, 512)
    x = block_conv2D(512, with_batch_norm=False)(x)
    # Shape = (1, 1, 512)
    x = layers.Flatten()(x)
    # Shape = (512)
    outputs = layers.Dense(512, activation='tanh')(x)
    # Define the model
    model = Model(inputs, outputs)
    
    return model


def make_generator_decoder_model():
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    inputs = layers.Input(shape=(512, ))
    x = layers.Dense(2*2*512, activation='tanh')(inputs)
    x = layers.Reshape((2,2,512))(x)
    # Shape = (2, 2, 512)
    x = block_conv2D_transpose(512, with_dropout=True)(x)
    # Shape = (4, 4, 512)
    x = block_conv2D_transpose(512, with_dropout=True)(x)
    # Shape = (8, 8, 512)
    x = block_conv2D_transpose(512)(x)
    # Shape = (16, 16, 512)
    x = block_conv2D_transpose(512)(x)
    # Shape = (32, 32, 512)
    x = block_conv2D_transpose(256)(x)
    # Shape = (64, 64, 256)
    x = block_conv2D_transpose(128)(x)
    # Shape = (128, 128, 128)
    x = block_conv2D_transpose(64)(x)
    # Shape = (256, 256, 64)
    # Last ouput 
    outputs = layers.Conv2DTranspose(3, kernel_size=4, strides=(1,1), activation="tanh", kernel_initializer=initializer, padding="same")(x)
    # Shape = (256, 256, 3)
    # Define the model
    model = Model(inputs, outputs)

    return model


def make_generator_encoder_decoder_model(encoder, decoder):
    inp = layers.Input((256, 256, 3))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    model = Model(inp, decoded)
    return model


def make_generator_unet_model():
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    inputs = layers.Input(shape=(256, 256, 3))
    res = []

    ### [First half of the network: downsampling inputs] ###
    
    # Entry block
    x = block_conv2D(64, with_batch_norm=False)(inputs)
    # Shape = (128, 128, 64)
    res.append(x)

    x = block_conv2D(128)(x)
    # Shape = (64, 64, 128)
    res.append(x)

    x = block_conv2D(256)(x)
    # Shape = (32, 32, 256)
    res.append(x)

    x = block_conv2D(512)(x)
    # Shape = (16, 16, 512)
    res.append(x)

    x = block_conv2D(512)(x)
    # Shape = (8, 8, 512)
    res.append(x)

    x = block_conv2D(512)(x)
    # Shape = (4, 4, 512)
    res.append(x)

    x = block_conv2D(512)(x)
    # Shape = (2, 2, 512)
    res.append(x)

    x = block_conv2D(512, with_batch_norm=False)(x)
    # Shape = (1, 1, 512) --> LATENT SPACE


    ### [Second half of the network: upsampling inputs] ###

    x = block_conv2D_transpose(512, with_dropout=True)(x)
    x = layers.Concatenate()([x, res[-1]])
    # Shape = (2, 2, 1024)

    x = block_conv2D_transpose(512, with_dropout=True)(x)
    x = layers.Concatenate()([x, res[-2]])
    # Shape = (4, 4, 1024)

    x = block_conv2D_transpose(512, with_dropout=True)(x)
    x = layers.Concatenate()([x, res[-3]])
    # Shape = (8, 8, 1024)

    x = block_conv2D_transpose(512)(x)
    x = layers.Concatenate()([x, res[-4]])
    # Shape = (16, 16, 1024)

    x = block_conv2D_transpose(256)(x)
    x = layers.Concatenate()([x, res[-5]])
    # Shape = (32, 32, 512)

    x = block_conv2D_transpose(128)(x)
    x = layers.Concatenate()([x, res[-6]])
    # Shape = (64, 64, 256)

    x = block_conv2D_transpose(64)(x)
    x = layers.Concatenate()([x, res[-7]])
    # Shape = (128, 128, 128)

    # Last ouput 
    outputs = layers.Conv2DTranspose(3, kernel_size=4, strides=(2,2), activation="tanh", kernel_initializer=initializer, padding="same")(x)
    # Shape = (256, 256, 3)

    # Define the model
    model = Model(inputs, outputs)

    return model


def downsample(filters, kernel_size=4, apply_batchnorm=True, alpha=0.3):
  initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
  result = Sequential()
  result.add(
      layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(layers.BatchNormalization())
  result.add(layers.LeakyReLU(alpha=alpha))

  return result


def make_discriminator_model(cgan_mode=True):
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    if cgan_mode:
        inputs = layers.Input(shape=[256, 256, 6])
    else:
        inputs = layers.Input(shape=[256, 256, 3])
    x = downsample(64, apply_batchnorm=False)(inputs)  # (128, 128, 64)
    x = downsample(128)(x)  # (64, 64, 128)
    x = downsample(256)(x)  # (32, 32, 256)
    x = downsample(512)(x)  # (16, 16, 512)
    x = downsample(512)(x)  # (8, 8, 512)
    x = downsample(512)(x)  # (4, 4, 512)
    x = layers.Flatten()(x) # (4*4*512)
    outputs = layers.Dense(1, kernel_initializer=initializer)(x)
    model = Model(inputs, outputs)

    return model

def make_patch_discriminator_model(cgan_mode=True):
    initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
    if cgan_mode:
        inputs = layers.Input(shape=[256, 256, 6])
    else:
        inputs = layers.Input(shape=[256, 256, 3])

    down1 = downsample(64, apply_batchnorm=False)(inputs)  # (batch_size, 128, 128, 64)
    down2 = downsample(128)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    outputs = layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
    model = Model(inputs, outputs)
    
    return model


def block_1_conv2D(filters, kernel_size, stride, activation='relu'):
    '''
        Return a block of layers consisting of a Conv2D-BatchNormal-LeakyRELu layer with
        convolution applying stride
    '''
    block = Sequential([
        layers.Conv2D(filters,
                      kernel_size,
                      strides=stride,
                      padding="same",
                      use_bias=True),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.1)
    ])

    return block


def block_1_conv2D_transpose(filters,
                             kernel_size,
                             stride,
                             activation='relu'):
    '''
        Return a block of layers consisting of a Conv2D-BatchNormal-LeakyRELu layer with
        convolution applying stride
    '''
    block = Sequential([
        layers.Conv2DTranspose(filters,
                               kernel_size,
                               strides=stride,
                               padding="same",
                               use_bias=True),
        layers.BatchNormalization(),
        layers.Activation(activation=activation)
    ])

    return block


def make_dummy_generator():

    inputs = layers.Input(shape=(256, 256, 3))

    # Entry block
    x = block_1_conv2D(64, 4, 1)(inputs)
    x = block_1_conv2D_transpose(64, 4, 1, activation='relu')(x)

    # Last ouput
    outputs = layers.Conv2DTranspose(3, 5, activation="tanh", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model


def make_dummy_discriminator(cgan_mode=False):
    model = Sequential()

    if cgan_mode:
        model.add(layers.Input(shape=(256, 256, 6)))
    else:
        model.add(layers.Input(shape=(256, 256, 3)))

    model.add(
        layers.Conv2D(64,
                      kernel_size=(5, 5),
                      strides=(4, 4),
                      padding='valid',
                      use_bias=False,
                      input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())

    model.add(layers.Dense(1))
    return model
