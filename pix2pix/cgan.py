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

"""
Main class for pix2pix project. Implement a full CGAN model that can be fit.
"""

class CGAN:
    def __init__(self, generator, discriminator=None):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = Adam(1e-4)
        self.disc_optimizer = Adam(1e-4)
        self.cross_entropy = CrossLoss(from_logits=False)
        self.l1 = L1Loss()
    
    
    #TODO add the possibility to add different metrics
    def compile(self, gen_optimizer, disc_optimizer, gen_metrics, disc_metrics):
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
    
    
    def discriminator_loss(self, real_output, fake_output):
        """Return the discriminator loss

        Args:
            real_output (tf.Tensor): Discriminator output
            fake_output ([type]): [description]

        Returns:
            [type]: [description]
        """
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    def update_discriminator_tracker(self, tracker, real_output, fake_output):
        tracker.update_state(ones_like(real_output), real_output)
        tracker.update_state(zeros_like(fake_output), fake_output)
        
    
    def generator_loss(self, fake_images=None, real_images=None, fake_output=None, loss_strategy='both'):
            #TODO with try/except
        assert loss_strategy in ['GAN', 'L1', 'both'], "Error: invalid type of loss. Should be 'GAN', 'L1' or 'both'"
        if loss_strategy == "GAN":
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            return fake_loss
        elif loss_strategy == "L1":
            L1_loss = self.l1(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)
            return L1_loss
        elif loss_strategy == 'both':
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            L1_loss = self.l1(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)
            return fake_loss + L1_loss
            
    def update_generator_mae(self, mae, fake_images, real_images):
        mae.update_state(real_images * 127.5 + 127.5, fake_images * 127.5 + 127.5)


    def update_generator_cross(self, cross, fake_output):
        cross.update_state(ones_like(fake_output), fake_output)
    
        
    @function
    def train_generator_step(self, paint, real_images, gen_mae):
        # Forward propagation
        with GradientTape() as gen_tape:
            fake_images = self.generator(paint, training=True)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images, loss_strategy='L1')

        # Compute gradients and apply to weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Track loss and metrics (loss = mae when the generator trains alone)
        self.update_generator_mae(gen_mae, fake_images, real_images)

    @function
    def train_discriminator_step(self, paint, real_images, disc_cross, disc_acc):
        # Forward propagation
        with GradientTape() as disc_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Compute gradients and apply to weights
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Track loss and metrics (loss = mae when the generator trains alone)
        self.update_discriminator_tracker(disc_cross, real_output, fake_output)
        self.update_discriminator_tracker(disc_acc, real_output, fake_output)

    @function
    def train_gan_step(self, paint, real_images, disc_cross, disc_acc, gen_cross, gen_mae):
        # Forward propagation
        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images, fake_output=fake_output, loss_strategy='GAN')
        
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
    
    
    def initialize_history(self):
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
        return history
      
      
    def update_history(self, history, epoch, cur_gen_train_mae, cur_gen_train_cross,
                   cur_disc_train_cross, cur_disc_train_acc):
        history.get('epoch_index', []).append(epoch+1)
        history.get('train', {}).get('gen_mae', []).append(cur_gen_train_mae)
        # If generator is training alone, then we're done by saving MAE as loss + metric
        if epoch < 1:
            history.get('train', {}).get('gen_loss', []).append(cur_gen_train_mae)
        else:
            #TODO: if generator's loss is L1 + crossentropy, we should add both
            history.get('train', {}).get('gen_loss', []).append(cur_gen_train_cross)
            history.get('train', {}).get('disc_loss', []).append(cur_disc_train_cross)
            history.get('train', {}).get('disc_acc', []).append(cur_disc_train_acc)
    
    
    def display_image(self, ax, sample_tensor):
        ax.imshow((sample_tensor * 127.5 + 127.5).numpy().astype('uint8'))
        ax.axis('off')
    
    
    def generate_and_save_images(self, model, epoch, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, trackers_to_display):
        display.clear_output(wait=True)
        #TODO: Use next_iter to avoid iterating upon the whole datasets ?
        X_train = [X for X in iter(X_ds_train)]
        Y_train = [Y for Y in iter(Y_ds_train)]
        X_val = [X for X in iter(X_ds_val)]
        Y_val = [Y for Y in iter(Y_ds_val)]

        index_batch_train = random.randint(0, len(X_train) - 1)
        index_batch_val = random.randint(0, len(X_val) - 1)
        index_train = random.randint(0, X_train[index_batch_train].shape[0] - 1)
        index_val = random.randint(0, X_val[index_batch_val].shape[0] - 1)

        prediction_train = model(expand_dims(X_train[index_batch_train][index_train], axis=0), training=False)[0]
        prediction_val = model(expand_dims(X_val[index_batch_val][index_val], axis=0), training=False)[0]

        fig = plt.figure(constrained_layout=True, figsize=(6,6))
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, :])
        
        self.display_image(ax1, X_train[index_batch_train][index_train])
        ax1.set_title(label="Train sample \n Input")
        self.display_image(ax2, Y_train[index_batch_train][index_train])
        ax2.set_title(label="Ground truth")
        self.display_image(ax3, prediction_train)
        ax3.set_title(label=f"Output")
        self.display_image(ax4, X_val[index_batch_val][index_val])
        ax4.set_title(label="Val sample \n Input")
        self.display_image(ax5, Y_val[index_batch_val][index_val])
        ax5.set_title(label="Ground truth")
        self.display_image(ax6, prediction_val)
        ax6.set_title(label="Output")
        ax7.text(0, 1, trackers_to_display, ha='left')
        ax7.axis('off')

        fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
        
        
    def display_trackers(self, start_training, start_epoch, epoch, epoch_gen, epochs, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc):
        display_str = f'''
            Training Phase
            Epoch {epoch+1:5}/{epochs:5} 
            Elapsed time since training   {round(time.time()-start_training, 2):8}s 
                          since last epoch       {round(time.time()-start_epoch, 2):8}s
            '''
        # If generator is training alone, then we're done by saving MAE as loss + metric
        if epoch < epoch_gen:
            display_str += f'''
            Train set : Generator loss = {cur_gen_train_mae:0.2f}            Generator MAE = {cur_gen_train_mae:0.2f}
            '''
        # If generator + discriminator are training, save and output all the trackers
        else:
            display_str += f'''
            Train set : Generator loss = {cur_gen_train_mae:0.2f}            Generator MAE = {cur_gen_train_mae:0.2f}
            '''
            display_str += f'''
                        Discriminator loss = {cur_disc_train_cross:0.2f}     Discriminator accuracy = {cur_disc_train_acc:0.2f}
            '''
        return display_str
    
    def fit_generator(self, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, epochs, history=None):
        start_training = time.time()
        # INITIALIZING
        if history == None:
            history = self.initialize_history()
        # Define the trackers to track loss
        gen_train_mae = MeanAbsoluteError()

        # START TRAINING
        for epoch in range(epochs):
            start_epoch = time.time()
            # Reset trackers for loss and metrics
            gen_train_mae.reset_state()

            # LOOP THROUGH EACH BATCH
            for paint_batch, image_batch in zip(X_ds_train, Y_ds_train):
                self.train_generator_step(paint_batch, image_batch, gen_train_mae)

            # OUTPUT A RANDOM PREDICTION EVERY 5 EPOCHS
            self.generate_and_save_images(self.generator, epoch, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

            # OUTPUT AND SAVE TRACKERS AT EACH EPOCH
            cur_gen_train_mae = np.round(gen_train_mae.result().numpy())
            
            print(f'Time for epoch {epoch+1} is {round(time.time()-start_epoch)} sec')
            history['epoch_index'].append(epoch+1)
            history['train']['gen_mae'].append(cur_gen_train_mae)
            history['train']['gen_loss'].append(cur_gen_train_mae)
            print(f'Train set : Gen loss = {cur_gen_train_mae} ; Gen MAE = {cur_gen_train_mae}\n')

        # Generate after the final epoch
        self.generate_and_save_images(self.generator, epochs, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val)

        return history


    def fit_gan(self, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, epochs, epoch_gen, epoch_disc, history=None):
        start_training = time.time()
        # INITIALIZING
        if history == None:
            history = self.initialize_history()
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
            start_epoch = time.time()
            # Reset trackers for loss and metrics
            for tracker in tracker_list:
                tracker.reset_state()

            # LOOP THROUGH EACH BATCH
            for paint_batch, image_batch in zip(X_ds_train, Y_ds_train):
            # if epoch < epoch_gen, train the generator alone
                if epoch < epoch_gen:
                    self.train_generator_step(paint_batch, image_batch, gen_train_mae)
                # for epoch >= epoch_gen, train generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.train_discriminator_step(paint_batch, image_batch, disc_train_cross, disc_train_acc)
                else:
                    self.train_gan_step(paint_batch, image_batch, disc_train_cross, disc_train_acc, gen_train_cross, gen_train_mae)


            # OUTPUT AND SAVE IMAGES+TRACKERS AT EACH EPOCH
            cur_gen_train_mae = gen_train_mae.result().numpy()
            cur_gen_train_cross = gen_train_cross.result().numpy()
            cur_disc_train_cross = disc_train_cross.result().numpy()
            cur_disc_train_acc = disc_train_acc.result().numpy()

            self.update_history(history, epoch, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc)
            trackers_to_display = self.display_trackers(start_training, start_epoch, epoch, epoch_gen, epochs, cur_gen_train_mae, cur_gen_train_cross, cur_disc_train_cross, cur_disc_train_acc)

            self.generate_and_save_images(self.generator, epoch, epochs, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, trackers_to_display)

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
    
    model = CGAN(generator, discriminator)
    
    epochs = 10
    epoch_gen = 2
    epoch_disc = 2
    history = model.fit_gan(paint_ds_train, real_ds_train, paint_ds_val, real_ds_val, epochs, epoch_gen, epoch_disc)