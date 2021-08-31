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
from pix2pix.utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow import concat

"""
Main class for pix2pix project. Implement a full CGAN model that can be fit.
"""

class CGAN:
    def __init__(self, generator, discriminator=None, cgan_mode=False):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = Adam(1e-4)
        self.disc_optimizer = Adam(1e-4)
        self.cross_entropy = CrossLoss(from_logits=False)
        self.l1 = L1Loss()
        self.random_sample = True
        self.disc_threshold = 0
        self.cgan_mode = cgan_mode

    #TODO add the possibility to add different metrics
    def compile(self, gen_optimizer, disc_optimizer, gen_metrics, disc_metrics):
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def update_discriminator_loss(self, loss, real_output, fake_output):
        loss.update_state(ones_like(real_output), real_output)
        loss.update_state(zeros_like(fake_output), fake_output)


    def update_discriminator_accuracy(self, acc, real_output, fake_output):
        real_proba = [0 if x <= self.disc_threshold else 1 for x in real_output]
        fake_proba = [0 if x <= self.disc_threshold else 1 for x in fake_output]
        acc.update_state(ones_like(real_output), np.array(real_proba))
        acc.update_state(zeros_like(fake_output), np.array(fake_proba))


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


    def train_generator_step(self, paint_images, real_images,
                             loss_tracker_train_gen, loss_tracker_train_disc,
                             metric_tracker_train_gen,
                             metric_tracker_train_disc, l1_lambda):
        # Forward propagation
        with GradientTape() as gen_tape:
            fake_images = self.generator(paint_images, training=True)
            if self.cgan_mode: # in cgan mode the paint images are also used as input of the discriminator
                disc_real_input = concat([paint_images, real_images], axis=3) # stack images on the column axis, i.e. the same we use to split them beforehand
                disc_fake_input = concat([paint_images, fake_images], axis=3)
            else:
                disc_real_input = real_images
                disc_fake_input = fake_images
            real_output = self.discriminator(disc_real_input, training=True)
            fake_output = self.discriminator(disc_fake_input, training=True)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images,
                                           l1_lambda=l1_lambda, loss_strategy='L1')

        # Compute gradients and apply to weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Track loss and metrics
        # For discriminators
        self.update_discriminator_loss(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_train_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images)


    def train_discriminator_step(self, paint_images, real_images,
                                 loss_tracker_train_gen, loss_tracker_train_disc,
                                 metric_tracker_train_gen, metric_tracker_train_disc):
        # Forward propagation
        with GradientTape() as disc_tape:
            fake_images = self.generator(paint_images, training=True)
            if self.cgan_mode: # in cgan mode the paint images are also used as input of the discriminator
                disc_real_input = concat([paint_images, real_images], axis=3) # stack images on the column axis, i.e. the same we use to split them beforehand
                disc_fake_input = concat([paint_images, fake_images], axis=3)
            else:
                disc_real_input = real_images
                disc_fake_input = fake_images
            real_output = self.discriminator(disc_real_input, training=True)
            fake_output = self.discriminator(disc_fake_input, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Compute gradients and apply to weights
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Track loss and metrics
        # For discriminators
        self.update_discriminator_loss(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_train_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images)


    def train_gan_step(self, paint_images, real_images,
                       loss_tracker_train_gen, loss_tracker_train_disc,
                       metric_tracker_train_gen, metric_tracker_train_disc,
                       l1_lambda):
        # Forward propagation
        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            fake_images = self.generator(paint_images, training=True)
            if self.cgan_mode:  # in cgan mode the paint images are also used as input of the discriminator
                disc_real_input = concat([paint_images, real_images], axis=3)  # stack images on the column axis, i.e. the same we use to split them beforehand
                disc_fake_input = concat([paint_images, fake_images], axis=3)
            else:
                disc_real_input = real_images
                disc_fake_input = fake_images
            real_output = self.discriminator(disc_real_input, training=True)
            fake_output = self.discriminator(disc_fake_input, training=True)
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
        self.update_discriminator_loss(loss_tracker_train_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_train_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_train_gen, fake_output)
        self.update_generator_mae(metric_tracker_train_gen, fake_images, real_images)


    def val_generator_step(self, paint_images, real_images,
                           loss_tracker_val_gen, loss_tracker_val_disc,
                           metric_tracker_val_gen, metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint_images, training=True)
        if self.cgan_mode: # in cgan mode the paint images are also used as input of the discriminator
            disc_real_input = concat([paint_images, real_images], axis=3) # stack images on the column axis, i.e. the same we use to split them beforehand
            disc_fake_input = concat([paint_images, fake_images], axis=3)
        else:
            disc_real_input = real_images
            disc_fake_input = fake_images
        real_output = self.discriminator(disc_real_input, training=True)
        fake_output = self.discriminator(disc_fake_input, training=True)

        # Track loss and metrics
        # For discriminators
        self.update_discriminator_loss(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_val_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images)


    def val_discriminator_step(self, paint_images, real_images,
                               loss_tracker_val_gen, loss_tracker_val_disc,
                               metric_tracker_val_gen, metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint_images, training=True)
        if self.cgan_mode:  # in cgan mode the paint images are also used as input of the discriminator
            disc_real_input = concat([paint_images, real_images], axis=3)  # stack images on the column axis, i.e. the same we use to split them beforehand
            disc_fake_input = concat([paint_images, fake_images], axis=3)
        else:
            disc_real_input = real_images
            disc_fake_input = fake_images

        real_output = self.discriminator(disc_real_input, training=True)
        fake_output = self.discriminator(disc_fake_input, training=True)
        # Track loss and metrics
        # For discriminators
        self.update_discriminator_loss(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_val_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images)


    def val_gan_step(self, paint_images, real_images, loss_tracker_val_gen,
                     loss_tracker_val_disc, metric_tracker_val_gen,
                     metric_tracker_val_disc):
        # Forward propagation
        fake_images = self.generator(paint_images, training=True)
        if self.cgan_mode:  # in cgan mode the paint images are also used as input of the discriminator
            disc_real_input = concat([paint_images, real_images], axis=3)  # stack images on the column axis, i.e. the same we use to split them beforehand
            disc_fake_input = concat([paint_images, fake_images], axis=3)
        else:
            disc_real_input = real_images
            disc_fake_input = fake_images

        real_output = self.discriminator(disc_real_input, training=True)
        fake_output = self.discriminator(disc_fake_input, training=True)
        # Track loss and metrics
        # For discriminators
        self.update_discriminator_loss(loss_tracker_val_disc, real_output, fake_output)
        self.update_discriminator_accuracy(metric_tracker_val_disc, real_output, fake_output)
        # For generators
        self.update_generator_cross(loss_tracker_val_gen, fake_output)
        self.update_generator_mae(metric_tracker_val_gen, fake_images, real_images)


    def initialize_history(self, epochs=0, epoch_gen=0, epoch_disc=0, l1_lambda=0):
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


    def generate_and_save_images(self, model, epoch, train_ds, val_ds, trackers_to_display, display_tracker=True, display_plot=True):
        display.clear_output(wait=True)
        #TODO: Use next_iter to avoid iterating upon the whole datasets ?
        train_list = [(paint, real) for paint, real in iter(train_ds)]
        val_list = [(paint, real) for paint, real in iter(val_ds)]

        if self.random_sample :
            index_batch_train = random.randint(0, len(train_list) - 1)
            index_batch_val = random.randint(0, len(val_list) - 1)
            index_train = random.randint(0, train_list[index_batch_train][0].shape[0] - 1)
            index_val = random.randint(0, val_list[index_batch_val][0].shape[0] - 1)
        else:
            index_batch_train = 0
            index_batch_val = 0
            index_train = 0
            index_val = 0

        prediction_train = model(expand_dims(
            train_list[index_batch_train][0][index_train], axis=0),
                                 training=False)
        prediction_val = model(expand_dims(
            val_list[index_batch_val][0][index_val], axis=0),
                               training=False)

        fig = plt.figure(constrained_layout=True, figsize=(18,9))

        gs = fig.add_gridspec(5, 6)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax3 = fig.add_subplot(gs[0:2, 2])
        ax4 = fig.add_subplot(gs[2:4, 0])
        ax5 = fig.add_subplot(gs[2:4, 1])
        ax6 = fig.add_subplot(gs[2:4, 2])
        ax7 = fig.add_subplot(gs[4, :3])
        ax8 = fig.add_subplot(gs[0:2, 3:])
        ax9 = fig.add_subplot(gs[2:4, 3:])

        self.display_image(ax1, train_list[index_batch_train][0][index_train])
        ax1.set_title(label="Train sample \n Input")
        self.display_image(ax2, prediction_train[0])
        ax2.set_title(label=f"Output")
        self.display_image(ax3, train_list[index_batch_train][1][index_train])
        ax3.set_title(label="Ground truth")
        self.display_image(ax4, val_list[index_batch_val][0][index_val])
        ax4.set_title(label="Val sample \n Input")
        self.display_image(ax5, prediction_val[0])
        ax5.set_title(label="Output")
        self.display_image(ax6, val_list[index_batch_val][1][index_val])
        ax6.set_title(label="Ground truth")
        ax7.text(0, 1, trackers_to_display, ha='left', size='medium')
        ax7.axis('off')
        
        plot_last_n_epochs(ax8, self.history, set_name='train', show_label=False)
        plot_last_n_epochs(ax9, self.history, set_name='val', show_label=True)
        
        fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


    def display_trackers(self, start_training, start_epoch, epoch, epoch_gen, epoch_disc, epochs, res_trackers_dict):
        if epoch < epoch_gen:
            display_str = f"\n Training Phase - Generator ({epoch+1}/{epoch_gen})\n"
        elif epoch < epoch_gen + epoch_disc:
            display_str = f"\n Training Phase - Discriminator ({epoch+1-epoch_gen}/{epoch_disc})\n"
        else:
            display_str = f"\n Training Phase : GAN ({epoch+1-epoch_gen-epoch_disc}/{epochs-epoch_disc-epoch_gen})\n"

        display_str += f'''
            Epoch {epoch+1:3}/{epochs:3}
            Elapsed time
                        - since training      {round(time.time()-start_training, 2):8}s
                        - since last epoch    {round(time.time()-start_epoch, 2):8}s
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
                            Discriminator loss = {res_trackers_dict['loss_tracker_train_disc']:0.2f}         Discriminator accuracy = {res_trackers_dict['metric_tracker_train_disc']:0.2f}

            Val set :   Generator GAN loss = {res_trackers_dict['loss_tracker_val_gen']:0.2f}        Generator MAE = {res_trackers_dict['metric_tracker_val_gen']:0.2f}
                            Discriminator loss = {res_trackers_dict['loss_tracker_val_disc']:0.2f}         Discriminator accuracy = {res_trackers_dict['metric_tracker_val_disc']:0.2f}
            '''
        return display_str


    def fit(self,
            train_ds=None,
            val_ds=None,
            epochs=0,
            epoch_gen=0,
            epoch_disc=0,
            l1_lambda=0):

        # ==== INITIALIZING ====
        start_training = time.time()
        self.initialize_history(epochs=epochs,
                                epoch_gen=epoch_gen,
                                epoch_disc=epoch_disc,
                                l1_lambda=l1_lambda)
        # Define the trackers to track loss ..
        loss_tracker_train_gen = BinaryCrossentropy(
            name='loss_tracker_train_gen')
        loss_tracker_train_disc = BinaryCrossentropy(
            name='loss_tracker_train_disc')
        loss_tracker_val_gen = BinaryCrossentropy(name='loss_tracker_val_gen')
        loss_tracker_val_disc = BinaryCrossentropy(
            name='loss_tracker_val_disc')
        # .. and metrics
        metric_tracker_train_gen = MeanAbsoluteError(
            name='metric_tracker_train_gen')
        metric_tracker_train_disc = Accuracy(name='metric_tracker_train_disc')
        metric_tracker_val_gen = MeanAbsoluteError(
            name='metric_tracker_val_gen')
        metric_tracker_val_disc = Accuracy(name='metric_tracker_val_disc')
        # Define a list of all the trackers, to ease their reset at each epoch
        trackers_list = [
            loss_tracker_train_gen, loss_tracker_train_disc,
            metric_tracker_train_gen, metric_tracker_train_disc,
            loss_tracker_val_gen, loss_tracker_val_disc,
            metric_tracker_val_gen, metric_tracker_val_disc
        ]
        # Define a list with all trackers' name as String
        trackers_name_list = [tracker.name for tracker in trackers_list]

        # ==== START FITING =====
        for epoch in range(epochs):
            start_epoch = time.time()
            # Reset trackers for loss and metrics
            for tracker in trackers_list:
                tracker.reset_state()

            # === TRAINING PHASE ON EACH BATCH ===
            for paint_train_batch, image_train_batch in train_ds:
                paint_train_batch, image_train_batch = random_jitter(
                    paint_train_batch, image_train_batch)
                # if epoch < epoch_gen, train the generator alone
                if epoch < epoch_gen:
                    self.train_generator_step(
                        paint_train_batch, image_train_batch,
                        loss_tracker_train_gen, loss_tracker_train_disc,
                        metric_tracker_train_gen, metric_tracker_train_disc,
                        l1_lambda)
                # for epoch >= epoch_gen, validate generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.train_discriminator_step(paint_train_batch,
                                                  image_train_batch,
                                                  loss_tracker_train_gen,
                                                  loss_tracker_train_disc,
                                                  metric_tracker_train_gen,
                                                  metric_tracker_train_disc)
                else:
                    self.train_gan_step(paint_train_batch, image_train_batch,
                                        loss_tracker_train_gen,
                                        loss_tracker_train_disc,
                                        metric_tracker_train_gen,
                                        metric_tracker_train_disc, l1_lambda)

            # === VALIDATION PHASE ON EACH BATCH ===
            for paint_val_batch, image_val_batch in val_ds:
                # if epoch < epoch_gen, validate the generator alone
                if epoch < epoch_gen:
                    self.val_generator_step(paint_val_batch, image_val_batch,
                                            loss_tracker_val_gen,
                                            loss_tracker_val_disc,
                                            metric_tracker_val_gen,
                                            metric_tracker_val_disc)
                # for epoch >= epoch_gen, validate generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.val_discriminator_step(paint_val_batch,
                                                image_val_batch,
                                                loss_tracker_val_gen,
                                                loss_tracker_val_disc,
                                                metric_tracker_val_gen,
                                                metric_tracker_val_disc)
                else:
                    self.val_gan_step(paint_val_batch, image_val_batch,
                                      loss_tracker_val_gen,
                                      loss_tracker_val_disc,
                                      metric_tracker_val_gen,
                                      metric_tracker_val_disc)

            # === OUTPUT AND SAVE IMAGES+TRACKERS AT EACH EPOCH ===
            # Create a list with all the trackers' results
            res_trackers_list = [
                tracker.result().numpy() for tracker in trackers_list
            ]
            # Then create a dict with trackers' names as keys and trackers' results as values
            res_trackers_dict = dict(zip(trackers_name_list,
                                         res_trackers_list))

            self.update_history(start_training, start_epoch, epoch,
                                res_trackers_dict)
            trackers_to_display = self.display_trackers(
                start_training, start_epoch, epoch, epoch_gen, epoch_disc,
                epochs, res_trackers_dict)
            self.generate_and_save_images(self.generator, epoch,
                                                  train_ds, val_ds,
                                                  trackers_to_display)




if __name__ == "__main__":
    train, val, test = get_dataset(host='local',
                                   dataset='facades',
                                   batch_size=32)
    generator = make_dummy_generator()
    discriminator = make_dummy_discriminator(cgan_mode=True)
    cgan = CGAN(generator, discriminator, cgan_mode=True)

    # paint_batch, real_batch = next(iter(train))
    # cgan.train_gan_step(paint_batch, real_batch, None, None, None, None, 100)

    #cgan.generate_and_save_images_from(generator, 10, train, val,None)

    cgan.fit(train_ds=train,
              val_ds=val,
              epochs=10, epoch_gen=1, epoch_disc=1,
              l1_lambda=100)
