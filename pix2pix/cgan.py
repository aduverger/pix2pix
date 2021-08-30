from tensorflow.keras.losses import BinaryCrossentropy as CrossLoss
from tensorflow.keras.losses import MeanAbsoluteError as L1Loss
from tensorflow import ones_like, zeros_like, GradientTape, function, expand_dims
from tensorflow.keras.metrics import BinaryCrossentropy, MeanAbsoluteError, Accuracy
from IPython import display
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from tensorflow.python.ops.gen_math_ops import Mean
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
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    def update_discriminator_tracker(self, tracker, real_output, fake_output):
        tracker.update_state(ones_like(real_output), real_output)
        tracker.update_state(zeros_like(fake_output), fake_output)
        
    
    def generator_loss(self, fake_images=None, real_images=None, fake_output=None, l1_lambda=100, loss_strategy='both'):
            #TODO with try/except
        assert loss_strategy in ['GAN', 'L1', 'both'], "Error: invalid type of loss. Should be 'GAN', 'L1' or 'both'"
        if loss_strategy == "GAN":
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            return fake_loss
        elif loss_strategy == "L1":
            L1_loss = self.l1(real_images, fake_images)
            return L1_loss
        elif loss_strategy == 'both':
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            L1_loss = self.l1(real_images, fake_images)
            return fake_loss + l1_lambda*L1_loss
            
    def update_generator_mae(self, mae, fake_images, real_images):
        mae.update_state(real_images, fake_images)


    def update_generator_cross(self, cross, fake_output):
        cross.update_state(ones_like(fake_output), fake_output)
    
        
    @function
    def train_generator_step(self, paint, real_images,
                             loss_tracker_train_gen, loss_tracker_train_disc,
                             metric_tracker_train_gen, metric_tracker_train_disc,
                             l1_lambda):
        # Forward propagation
        with GradientTape() as gen_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images,
                                           l1_lambda=l1_lambda, loss_strategy='L1')

        # Compute gradients and apply to weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_train_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images)


    @function
    def train_discriminator_step(self, paint, real_images,
                                 loss_tracker_train_gen, loss_tracker_train_disc,
                                 metric_tracker_train_gen, metric_tracker_train_disc):
        # Forward propagation
        with GradientTape() as disc_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Compute gradients and apply to weights
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_train_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images) 


    @function
    def train_gan_step(self, paint, real_images,
                       loss_tracker_train_gen, loss_tracker_train_disc,
                       metric_tracker_train_gen, metric_tracker_train_disc,
                       l1_lambda):
        # Forward propagation
        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            fake_images = self.generator(paint, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images,
                                           fake_output=fake_output, l1_lambda=l1_lambda, loss_strategy='both')
        
        # Compute gradients and apply to weights
            # For discriminators
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            # For generators
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_train_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images)    
    
    
    @function
    def val_generator_step(self, paint, real_images,
                           loss_tracker_val_gen, loss_tracker_val_disc,
                           metric_tracker_val_gen, metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint, training=True)
        real_output = self.discriminator(real_images, training=True)
        fake_output = self.discriminator(fake_images, training=True)

        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_val_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images)
    

    @function
    def val_discriminator_step(self, paint, real_images,
                               loss_tracker_val_gen, loss_tracker_val_disc,
                               metric_tracker_val_gen, metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint, training=True)
        real_output = self.discriminator(real_images, training=True)
        fake_output = self.discriminator(fake_images, training=True)

        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_val_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images) 


    @function
    def val_gan_step(self, paint, real_images,
                     loss_tracker_val_gen, loss_tracker_val_disc,
                     metric_tracker_val_gen, metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint, training=True)
        real_output = self.discriminator(real_images, training=True)
        fake_output = self.discriminator(fake_images, training=True)
                
        # Track loss and metrics
            # For discriminators
        self.update_discriminator_tracker(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_tracker(metric_tracker_val_disc, real_output, fake_output)
            # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images)    


    def initialize_history(self, epochs=0, epoch_gen=0, epoch_disc=0, l1_lambda=0):
        #TODO : Add epoch, epoch_gen, epoch_disc, l1_lambda
        history = {
            'epochs': epochs,
            'epoch_gen': epoch_gen,
            'epoch_disc': epoch_disc,
            'l1_lambda': l1_lambda,
            'epoch_index': [],
            'time_epoch': [],
            'time_cumulative': [],
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
        self.history = history
      
      
    def update_history(self, start_training, start_epoch, epoch, res_trackers_dict):
        self.history.get('epoch_index', []).append(epoch+1)
        self.history.get('time_epoch', []).append(time.time()-start_epoch)
        self.history.get('time_cumulative', []).append(time.time()-start_training)
        
        self.history.get('train', {}).get('gen_loss', []).append(res_trackers_dict['loss_tracker_train_gen'])
        self.history.get('train', {}).get('gen_mae', []).append(res_trackers_dict['metric_tracker_train_gen'])
        self.history.get('train', {}).get('disc_loss', []).append(res_trackers_dict['loss_tracker_train_disc'])
        self.history.get('train', {}).get('disc_acc', []).append(res_trackers_dict['metric_tracker_train_disc'])
        
        self.history.get('val', {}).get('gen_loss', []).append(res_trackers_dict['loss_tracker_val_gen'])
        self.history.get('val', {}).get('gen_mae', []).append(res_trackers_dict['metric_tracker_val_gen'])
        self.history.get('val', {}).get('disc_loss', []).append(res_trackers_dict['loss_tracker_val_disc'])
        self.history.get('val', {}).get('disc_acc', []).append(res_trackers_dict['metric_tracker_val_disc'])
    
    
    def display_image(self, ax, sample_tensor):
        ax.imshow((sample_tensor * 127.5 + 127.5).numpy().astype('uint8'))
        ax.axis('off')
    
    
    def generate_and_save_images(self, model, epoch, X_ds_train, Y_ds_train, X_ds_val, Y_ds_val, trackers_to_display, random_sample=False):
        display.clear_output(wait=True)
        #TODO: Use next_iter to avoid iterating upon the whole datasets ?
        X_train = [X for X in iter(X_ds_train)]
        Y_train = [Y for Y in iter(Y_ds_train)]
        X_val = [X for X in iter(X_ds_val)]
        Y_val = [Y for Y in iter(Y_ds_val)]
        
        if random_sample == False :
            index_batch_train = random.randint(0, len(X_train) - 1)
            index_batch_val = random.randint(0, len(X_val) - 1)
            index_train = random.randint(0, X_train[index_batch_train].shape[0] - 1)
            index_val = random.randint(0, X_val[index_batch_val].shape[0] - 1)
        else:
            index_batch_train = 0
            index_batch_val = 0
            index_train = 0
            index_val = 0

        prediction_train = model(expand_dims(X_train[index_batch_train][index_train], axis=0), training=False)[0]
        prediction_val = model(expand_dims(X_val[index_batch_val][index_val], axis=0), training=False)[0]

        fig = plt.figure(constrained_layout=True, figsize=(9,6))
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
        ax7.text(0, 1, trackers_to_display, ha='left', size='medium')
        ax7.axis('off')
        fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


    def display_trackers(self, start_training, start_epoch, epoch, epoch_gen, epoch_disc, epochs, res_trackers_dict):
        if epoch < epoch_gen:
            display_str = f"\n Training Phase : Generator leveling up ({epoch+1}/{epoch_gen})\n"
        elif epoch < epoch_gen + epoch_disc:
            display_str = f"\n Training Phase : Discriminator leveling up ({epoch+1-epoch_gen}/{epoch_disc})\n"
        else:
            display_str = f"\n Training Phase : GAN leveling up\n"
        
        display_str += f'''
            Epoch {epoch+1:3}/{epochs:3} 
            Elapsed time since training         {round(time.time()-start_training, 2):8}s 
                                  since last epoch    {round(time.time()-start_epoch, 2):8}s
            '''
        # If generator is training alone, its loss = mae
        if epoch < epoch_gen:
            display_str += f'''
            Train set : Generator L1 loss = {res_trackers_dict['metric_tracker_train_gen']:0.2f}            Generator MAE = {res_trackers_dict['metric_tracker_train_gen']:0.2f}
            
            Val set    : Generator L1 loss = {res_trackers_dict['metric_tracker_val_gen']:0.2f}            Generator MAE = {res_trackers_dict['metric_tracker_val_gen']:0.2f}
    
            '''
        else:
            display_str += f'''
            Train set : Generator GAN loss = {res_trackers_dict['loss_tracker_train_gen']:0.2f}        Generator MAE = {res_trackers_dict['metric_tracker_train_gen']:0.2f}
                            Discriminator loss = {res_trackers_dict['loss_tracker_train_disc']:0.2f}     Discriminator accuracy = {res_trackers_dict['metric_tracker_train_disc']:0.2f}
            
            Val set :   Generator GAN loss = {res_trackers_dict['loss_tracker_val_gen']:0.2f}        Generator MAE = {res_trackers_dict['metric_tracker_val_gen']:0.2f}
                            Discriminator loss = {res_trackers_dict['loss_tracker_val_disc']:0.2f}     Discriminator accuracy = {res_trackers_dict['metric_tracker_val_disc']:0.2f}
            '''
        return display_str
    

    def fit(self, X_ds_train=None, Y_ds_train=None,
            X_ds_val=None, Y_ds_val=None,
            epochs=0, epoch_gen=0, epoch_disc=0,
            l1_lambda=0):
        
        # ==== INITIALIZING ====
        start_training = time.time()
        self.initialize_history(epochs=epochs, epoch_gen=epoch_gen, epoch_disc=epoch_disc, l1_lambda=l1_lambda)
        # Define the trackers to track loss ..
        loss_tracker_train_gen = BinaryCrossentropy(name='loss_tracker_train_gen')
        loss_tracker_train_disc = BinaryCrossentropy(name='loss_tracker_train_disc')
        loss_tracker_val_gen = BinaryCrossentropy(name='loss_tracker_val_gen')
        loss_tracker_val_disc = BinaryCrossentropy(name='loss_tracker_val_disc')
        # .. and metrics
        metric_tracker_train_gen = MeanAbsoluteError(name='metric_tracker_train_gen')
        metric_tracker_train_disc = Accuracy(name='metric_tracker_train_disc')
        metric_tracker_val_gen = MeanAbsoluteError(name='metric_tracker_val_gen')
        metric_tracker_val_disc = Accuracy(name='metric_tracker_val_disc')
        # Define a list of all the trackers, to ease their reset at each epoch
        trackers_list = [loss_tracker_train_gen, loss_tracker_train_disc,
                        metric_tracker_train_gen, metric_tracker_train_disc,
                        loss_tracker_val_gen, loss_tracker_val_disc,
                        metric_tracker_val_gen, metric_tracker_val_disc]
        # Define a list with all trackers' name as String
        trackers_name_list = [tracker.name for tracker in trackers_list]
        
        # ==== START FITING =====
        for epoch in range(epochs):
            start_epoch = time.time()
            # Reset trackers for loss and metrics
            for tracker in trackers_list:
                tracker.reset_state()

            # === TRAINING PHASE ON EACH BATCH ===
            for paint_train_batch, image_train_batch in zip(X_ds_train, Y_ds_train):
                # if epoch < epoch_gen, train the generator alone
                if epoch < epoch_gen:
                    self.train_generator_step(
                        paint_train_batch, image_train_batch,
                        loss_tracker_train_gen, loss_tracker_train_disc,
                        metric_tracker_train_gen, metric_tracker_train_disc,
                        l1_lambda
                        )
                # for epoch >= epoch_gen, validate generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.train_discriminator_step(
                        paint_train_batch, image_train_batch,
                        loss_tracker_train_gen, loss_tracker_train_disc,
                        metric_tracker_train_gen, metric_tracker_train_disc
                        )
                else:
                    self.train_gan_step(
                        paint_train_batch, image_train_batch,
                        loss_tracker_train_gen, loss_tracker_train_disc,
                        metric_tracker_train_gen, metric_tracker_train_disc,
                        l1_lambda
                        )

            # === VALIDATION PHASE ON EACH BATCH ===
            for paint_val_batch, image_val_batch in zip(X_ds_val, Y_ds_val):
                # if epoch < epoch_gen, validate the generator alone
                if epoch < epoch_gen:
                    self.val_generator_step(
                        paint_val_batch, image_val_batch,
                        loss_tracker_val_gen, loss_tracker_val_disc,
                        metric_tracker_val_gen, metric_tracker_val_disc
                        )
                # for epoch >= epoch_gen, validate generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.val_discriminator_step(
                        paint_val_batch, image_val_batch,
                        loss_tracker_val_gen, loss_tracker_val_disc,
                        metric_tracker_val_gen, metric_tracker_val_disc
                        )
                else:
                    self.val_gan_step(
                        paint_val_batch, image_val_batch,
                        loss_tracker_val_gen, loss_tracker_val_disc,
                        metric_tracker_val_gen, metric_tracker_val_disc
                        )

            # === OUTPUT AND SAVE IMAGES+TRACKERS AT EACH EPOCH ===
                # Create a list with all the trackers' results
            res_trackers_list = [tracker.result().numpy() for tracker in trackers_list]
                # Then create a dict with trackers' names as keys and trackers' results as values
            res_trackers_dict = dict(zip(trackers_name_list, res_trackers_list))
            
            self.update_history(start_training, start_epoch, epoch, res_trackers_dict)
            trackers_to_display = self.display_trackers(start_training, start_epoch, epoch, epoch_gen,
                                                        epoch_disc, epochs, res_trackers_dict)
            self.generate_and_save_images(self.generator, epoch, X_ds_train, Y_ds_train,
                                          X_ds_val, Y_ds_val, trackers_to_display)


if __name__ == "__main__":
    paint_ds_train, paint_ds_val, paint_ds_test, real_ds_train, real_ds_val, real_ds_test = \
                                                                    get_facades_datasets(host='local', batch_size=1)
    latent_dim = 100
    encoder = make_generator_encoder_model(latent_dim)
    decoder = make_generator_decoder_model(latent_dim)
    generator = make_generator_autoencoder_model(encoder, decoder)
    discriminator = make_discriminator_model()
    model = CGAN(generator, discriminator)
    
    model.fit(X_ds_train=paint_ds_train, Y_ds_train=real_ds_train,
              X_ds_val=paint_ds_val, Y_ds_val=real_ds_val,
              epochs=10, epoch_gen=1, epoch_disc=1,
              l1_lambda=100)