from tensorflow.keras.losses import BinaryCrossentropy as CrossLoss
from tensorflow.keras.losses import MeanAbsoluteError as L1Loss
from tensorflow import ones_like, zeros_like, GradientTape, function, expand_dims
from tensorflow.keras.metrics import BinaryCrossentropy, MeanAbsoluteError, Accuracy
from IPython import display
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from pix2pix.data import *
from pix2pix.models import *
from tensorflow.keras.optimizers import Adam



class CGAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = Adam(1e-4)
        self.disc_optimizer = Adam(1e-4)
        self.cross_entropy = CrossLoss(from_logits=False)
        self.l1 = L1Loss()
    
    def compile(self, gen_optimizer, disc_optimizer, gen_metrics, disc_metrics):
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def update_discriminator_tracker(self, acc, real_output, fake_output):
        acc.update_state(ones_like(real_output), real_output)
        acc.update_state(zeros_like(fake_output), fake_output)
        
    def generator_loss(self, fake_images=None, real_images=None, fake_output=None, type_='both'):
        if type_ == "GAN":
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            return fake_loss
        elif type_ == "L1":
            L1_loss = self.l1(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)
            return L1_loss
        elif type_ == 'both':
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            L1_loss = self.l1(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)
            return fake_loss + L1_loss
        else:
            print("Error: invalid type of loss. Should be 'GAN', 'L1' or 'both'")
            
    def update_generator_mae(self, mae, fake_images, real_images):
        mae.update_state(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)
        
    def update_generator_cross(self, cross, fake_output):
        cross.update_state(ones_like(fake_output), fake_output)
    
    def display_image(self, ax, sample_tensor):
        ax.imshow((sample_tensor * 127.5 + 127.5).numpy().astype('uint8'))
        ax.axis('off')
        
    def generate_and_save_images(self, model, epoch, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val):
        display.clear_output(wait=True)
        
        X_train = [X for X in iter(X_ds_train)]
        Y_train = [Y for Y in iter(Y_ds_train)]
        X_val = [X for X in iter(X_ds_val)]
        Y_val = [Y for Y in iter(Y_ds_val)]
        
        index_batch_train = random.randint(0, len(X_train) - 1)
        index_batch_val = random.randint(0, len(X_val) - 1)
        # Get a random index to sample from train et val
        # TRAIN & VAL
        
        index_train = random.randint(0, X_train[index_batch_train].shape[0] - 1)
        index_val = random.randint(0, X_val[index_batch_val].shape[0] - 1)

        prediction_train = model(expand_dims(X_train[index_batch_train][index_train], axis=0), training=False)[0]
        prediction_val = model(expand_dims(X_val[index_batch_val][index_val], axis=0), training=False)[0]

        fig, axs = plt.subplots(2, 3, figsize=(10,7))
        self.display_image(axs[0,0], X_train[index_batch_train][index_train])
        axs[0,0].set_title(label="Train sample \n Input")
        self.display_image(axs[0,1], Y_train[index_batch_train][index_train])
        axs[0,1].set_title(label="Ground truth")
        self.display_image(axs[0,2], prediction_train)
        axs[0,2].set_title(label="Output")
        self.display_image(axs[1,0], X_val[index_batch_val][index_val])
        axs[1,0].set_title(label="Val sample \n Input")
        self.display_image(axs[1,1], Y_val[index_batch_val][index_val])
        axs[1,1].set_title(label="Ground truth")
        self.display_image(axs[1,2], prediction_val)
        axs[1,2].set_title(label="Output")

        fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
        
    #@function
    def train_generator_step(self, paint, real_images, gen_mae):
        # Forward propagation
        with GradientTape() as gen_tape:
            fake_images = self.generator(paint, training=True)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images, type_='L1')

        # Compute gradients and apply to weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Track loss and metrics (loss = mae when the generator trains alone)
        self.update_generator_mae(gen_mae, fake_images, real_images)

    #@function
    def train_gan_step(self, paint, real_images, disc_cross, disc_acc, gen_cross, gen_mae):
        # Forward propagation
        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images, fake_output=fake_output, type_='GAN')
        
        # Compute gradients and apply to weights
            # For discriminators
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            # For generators
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(disc_cross, real_output, fake_output)
        self.update_discriminator_tracker(disc_acc, real_output, fake_output)
            # For generators
        self.update_generator_cross(gen_cross, fake_output)
        self.update_generator_mae(gen_mae, fake_images, real_images)    
            
    def update_history(self, history, epoch, cur_gen_train_mae, cur_gen_train_cross,
                   cur_disc_train_cross, cur_disc_train_acc):
        history['epoch_index'].append(epoch+1)
        history['train']['gen_mae'].append(cur_gen_train_mae)
        # If generator is training alone, then we're done by saving MAE as loss + metric
        if epoch < 1:
            history['train']['gen_loss'].append(cur_gen_train_mae)
        else:
            #TODO: if generator's loss is L1 + crossentropy, we should add both
            history['train']['gen_loss'].append(cur_gen_train_cross)
            history['train']['disc_loss'].append(cur_disc_train_cross)
            history['train']['disc_acc'].append(cur_disc_train_acc)
    
    def display_trackers(self, start, epoch, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc):
        print(f'Time for epoch {epoch+1} is {round(time.time()-start, 2)} sec')
        # If generator is training alone, then we're done by saving MAE as loss + metric
        if epoch < 30:
            print(f'Train set : Gen loss = {cur_gen_train_mae} ; Gen MAE = {cur_gen_train_mae}\n')

        # If generator + discriminator are training, save and output all the trackers
        else:
            print(
                f'Train set : Gen loss = {cur_gen_train_cross} ; Gen MAE = {cur_gen_train_mae} ;\n \
                Disc loss = {cur_disc_train_cross} ; Disc acc = {cur_disc_train_acc}\n'
                )
            
    def train_generator(self, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, epochs, history=None):
        # INITIALIZING
        if history == None:
            history = {
                'epoch_index': [],
                'train': {
                    'gen_loss': [],
                    'disc_loss': [],
                    'gen_mae': [],
                    'disc_acc': [],
                },
                'val': {
                    'gen_loss': [],
                    'disc_loss': [],
                    'gen_mae': [],
                    'disc_acc': [],
                }
            }
        # Define the trackers to track loss
        gen_train_mae = MeanAbsoluteError()


        # START TRAINING
        for epoch in range(epochs):
            start = time.time()
            # Reset trackers for loss and metrics
            gen_train_mae.reset_state()

            # LOOP THROUGH EACH BATCH
            for paint_batch, image_batch in zip(X_ds_train, Y_ds_train):
                self.train_generator_step(paint_batch, image_batch, gen_train_mae)

            # OUTPUT A RANDOM PREDICTION EVERY 5 EPOCHS
            if epoch % 5 == 0:
                self.generate_and_save_images(self.generator, epoch, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

            # OUTPUT AND SAVE TRACKERS AT EACH EPOCH
            cur_gen_train_mae = np.round(gen_train_mae.result().numpy())
            
            print(f'Time for epoch {epoch+1} is {round(time.time()-start)} sec')
            history['epoch_index'].append(epoch+1)
            history['train']['gen_mae'].append(cur_gen_train_mae)
            history['train']['gen_loss'].append(cur_gen_train_mae)
            print(f'Train set : Gen loss = {cur_gen_train_mae} ; Gen MAE = {cur_gen_train_mae}\n')

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

        return history
    
    def train_gan(self, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, epochs, history=None):
        # INITIALIZING
        if history == None:
            history = {
                'epoch_index': [],
                'train': {
                    'gen_loss': [],
                    'disc_loss': [],
                    'gen_mae': [],
                    'disc_acc': [],
                },
                'val': {
                    'gen_loss': [],
                    'disc_loss': [],
                    'gen_mae': [],
                    'disc_acc': [],
                }
            }
        # Define the trackers to track loss ..
        gen_train_cross = BinaryCrossentropy()
        disc_train_cross = BinaryCrossentropy()
        # .. and metrics
        gen_train_mae = MeanAbsoluteError()
        disc_train_acc = Accuracy()
        # Define a list of all the trackers, to ease their reset at each epoch
        tracker_list = [gen_train_cross, disc_train_cross, gen_train_mae, disc_train_acc]


        # START TRAINING
        for epoch in range(epochs):
            start = time.time()
            # Reset trackers for loss and metrics
            for tracker in tracker_list:
                tracker.reset_state()

            # LOOP THROUGH EACH BATCH
            for paint_batch, image_batch in zip(X_ds_train, Y_ds_train):
            # if epoch < 30, train the generator alone
                if epoch < 5:
                    self.train_generator_step(paint_batch, image_batch, gen_train_mae)
                # for epoch >= 30, train generator + discriminator
                else:
                    self.train_gan_step(paint_batch, image_batch, disc_train_cross, disc_train_acc, gen_train_cross, gen_train_mae)


            # OUTPUT A RANDOM PREDICTION EVERY 5 EPOCHS
            if epoch % 5 == 0:
                self.generate_and_save_images(self.generator, epoch, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

            # OUTPUT AND SAVE TRACKERS AT EACH EPOCH
            cur_gen_train_mae = gen_train_mae.result().numpy()
            cur_gen_train_cross = gen_train_cross.result().numpy()
            cur_disc_train_cross = disc_train_cross.result().numpy()
            cur_disc_train_acc = disc_train_acc.result().numpy()

            self.update_history(history, epoch, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc)
            self.display_trackers(start, epoch, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc)

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

        return history
    
if __name__ == "__main__":
    paint_ds_train, paint_ds_val, paint_ds_test, real_ds_train, real_ds_val, real_ds_test = \
                                                                    get_facades_datasets(host='local')
    gen_optimizer = Adam(1e-4)
    disc_optimizer = Adam(1e-4)
    latent_dim = 100
    
    encoder = make_generator_encoder_model(latent_dim)
    decoder = make_generator_decoder_model(latent_dim)
    generator = make_generator_autoencoder_model(encoder, decoder)
    discriminator = make_discriminator_model()
    
    model = CGAN(generator, discriminator, gen_optimizer, disc_optimizer)
    
    epochs = 10
    history = model.train_gan(paint_ds_train, real_ds_train, paint_ds_val, real_ds_val, epochs)