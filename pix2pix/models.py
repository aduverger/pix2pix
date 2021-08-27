from tensorflow.keras import layers, Sequential, Model

def make_generator_encoder_model(latent_dim):
    encoder = Sequential()

    encoder.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(256, 256, 3)))
    encoder.add(layers.LeakyReLU(alpha=0.2))
    assert encoder.output_shape == (None, 128, 128, 64)  # Note: None is the batch size

    encoder.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 64, 64, 128)

    encoder.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 32, 32, 256)

    encoder.add(layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 16, 16, 512)

    encoder.add(layers.Conv2D(512, (4,4), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 8, 8, 512)

    encoder.add(layers.Conv2D(512, (3,3), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 4, 4, 512)

    encoder.add(layers.Conv2D(512, (2,2), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 2, 2, 512)

    encoder.add(layers.Conv2D(512, (2,2), strides=(2,2), padding='same'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.LeakyReLU(alpha=0.2))
    encoder.add(layers.Dropout(0.1))
    assert encoder.output_shape == (None, 1, 1, 512)

    encoder.add(layers.Flatten())
    assert encoder.output_shape == (None, 512)

    encoder.add(layers.Dense(latent_dim, activation='tanh'))

    return encoder

def make_generator_decoder_model(latent_dim):
    decoder = Sequential()

    decoder.add(layers.Dense(2*2*512, input_shape=(latent_dim,), activation='tanh'))

    decoder.add(layers.Reshape((2,2,512)))

    decoder.add(layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 4, 4, 512)

    decoder.add(layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 8, 8, 512)

    decoder.add(layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 16, 16, 512)

    decoder.add(layers.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 32, 32, 512)

    decoder.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 64, 64, 256)

    decoder.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 128, 128, 128)

    decoder.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.ReLU())
    assert decoder.output_shape == (None, 256, 256, 64)

    decoder.add(layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding='same', activation='tanh'))

    assert decoder.output_shape == (None, 256, 256, 3)

    return decoder

def make_generator_autoencoder_model(encoder, decoder):
    inp = layers.Input((256, 256, 3))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder

def make_discriminator_model():
  discriminator = Sequential()
  discriminator.add(layers.Conv2D(64, (5,5), strides=(1,1), input_shape=(256, 256,3), padding='same'))
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
  discriminator.add(layers.BatchNormalization())
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='same'))
  discriminator.add(layers.BatchNormalization())
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
  discriminator.add(layers.BatchNormalization())
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
  discriminator.add(layers.BatchNormalization())
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Conv2D(512, (4,4), strides=(2,2), padding='same'))
  discriminator.add(layers.BatchNormalization())
  discriminator.add(layers.LeakyReLU())

  discriminator.add(layers.Flatten())

  discriminator.add(layers.Dense(1, activation='sigmoid'))

  return discriminator