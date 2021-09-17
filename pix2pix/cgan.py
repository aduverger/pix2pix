from tensorflow.keras.losses import BinaryCrossentropy as CrossLoss
from tensorflow.keras.losses import MeanAbsoluteError as L1Loss
from tensorflow import ones_like, zeros_like, GradientTape
from tensorflow.keras.metrics import BinaryCrossentropy, MeanAbsoluteError, Accuracy
import time
import numpy as np
from tensorflow.python.ops.gen_math_ops import Mean
from pix2pix.data import *
from pix2pix.models import *
from pix2pix.display import *
from tensorflow.keras.optimizers import Adam
from tensorflow import concat

"""
Main class for pix2pix project. Implement a full cGAN model.
"""

class CGAN:
    def __init__(self, generator, discriminator=None, cgan_mode=False, random_sample=True):
        """"
        Args:
            generator (tf.keras.Model):
                    A tensorflow model for the generator.
                    Usually the GAN generator has a U-Net architecture.
            discriminator (tf.keras.Model):
                    A tensorflow model for the discriminator.
                    Usually the GAN discriminator has a Markovian architecture (PatchGAN) 
            cgan_mode (bool, optional):
                    Set to True if you want the discriminator to be condititional
                    (meaning it takes a concatenation of X and Y as input). Defaults to False.
            random_sample (bool, optional):
                    Set to False if you want the same sample from train and val to be displayed
                    and saved at each epoch. Usefull if you want to make GIF. Defaults to True.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = Adam(1e-4)
        self.disc_optimizer = Adam(1e-4)
        self.cross_entropy = CrossLoss(from_logits=True)
        self.history = self.initialize_history()
        self.l1 = L1Loss()
        self.random_sample = random_sample
        if not random_sample: # WORKS ONLY ON GOOGLE COLAB FOR NOW
            self.paint_train, self.real_train = \
                                load_and_split_image('/content/drive/MyDrive/pix2pix/datasets/resized/train/20.jpg')
            self.paint_val, self.real_val = \
                                load_and_split_image('/content/drive/MyDrive/pix2pix/datasets/resized/val/21.jpg')
        self.disc_threshold = 0
        self.cgan_mode = cgan_mode

    #TODO add the possibility to add different metrics
    def compile(self, gen_optimizer, disc_optimizer):
        """Change the optimizers of a cGAN.

        Args:
            gen_optimizer (tf.keras.optimizers): Generator optimizer
            disc_optimizer (tf.keras.optimizer): Discriminator optimizer
        """
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer


    def discriminator_loss(self, real_output, fake_output):
        """Compute the loss for the discriminator
        """
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def update_discriminator_loss(self, loss, real_output, fake_output):
        """Update the loss for the discriminator
        """
        loss.update_state(ones_like(real_output), real_output)
        loss.update_state(zeros_like(fake_output), fake_output)


    def update_discriminator_accuracy(self, acc, real_output, fake_output):
        """Update the accuracy for the discriminator
        """
        real_output_array = real_output.numpy()
        real_output_array = np.vectorize(lambda x: 0 if x < self.disc_threshold else 1)(real_output_array)

        fake_output_array = fake_output.numpy()
        fake_output_array = np.vectorize(lambda x: 0 if x < self.disc_threshold else 1)(fake_output_array)

        #real_proba = [0 if x <= self.disc_threshold else 1 for x in real_output]
        #fake_proba = [0 if x <= self.disc_threshold else 1 for x in fake_output]
        acc.update_state(ones_like(real_output), np.array(real_output_array))
        acc.update_state(zeros_like(fake_output), np.array(fake_output_array))


    def generator_loss(self, fake_images=None, real_images=None, fake_output=None, l1_lambda=100, loss_strategy='both'):
        """Compute the loss for the generator
        """
        #TODO with try/except
        assert loss_strategy in ['GAN', 'L1', 'both'], "Error: invalid type of loss. Should be 'GAN', 'L1' or 'both'"
        if loss_strategy == "GAN":
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            return fake_loss
        elif loss_strategy == "L1":
            L1_loss = l1_lambda*self.l1(real_images, fake_images)
            return L1_loss
        elif loss_strategy == 'both':
            fake_loss = self.cross_entropy(ones_like(fake_output), fake_output)
            L1_loss = self.l1(real_images, fake_images)
            return fake_loss + l1_lambda*L1_loss


    def update_generator_mae(self, mae, fake_images, real_images):
        """Update the MAE metric for the generator
        """
        mae.update_state(real_images, fake_images)


    def update_generator_cross(self, cross, fake_output):
        """Update the binary crossentropy metric for the generator
        """
        cross.update_state(ones_like(fake_output), fake_output)


    def train_generator_step(self, paint_images, real_images,
                             metric_tracker_train_gen, l1_lambda):
        # Forward propagation
        with GradientTape() as gen_tape:
            fake_images = self.generator(paint_images, training=True)
            gen_loss = self.generator_loss(fake_images=fake_images, real_images=real_images,
                                           l1_lambda=l1_lambda, loss_strategy='L1')

        # Compute gradients and apply to weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Track loss and metrics
        # For generators : at this phase, generator's loss = mae
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
                disc_real_input = concat([paint_images, real_images], axis=3)
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


    def val_generator_step(self, paint_images, real_images, metric_tracker_val_gen):
        # Forward propagation
        fake_images = self.generator(paint_images, training=True)

        # Track loss and metrics
        # For generators : at this phase, generator's loss = mae
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


    def initialize_history(self):
        history = {
            'epochs': [],
            'epoch_gen': [],
            'epoch_disc': [],
            'l1_lambda': [],
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
        return history


    def fit_history(self, epochs, epoch_gen, epoch_disc, l1_lambda):
        """ Update history at each .fit() with arguments
        """
        self.history.get('epochs', []).append(epochs)
        self.history.get('epoch_gen', []).append(epoch_gen)
        self.history.get('epoch_disc', []).append(epoch_disc)
        self.history.get('l1_lambda', []).append(l1_lambda)


    def update_history(self, start_training, start_epoch, epoch, res_trackers_dict):
        """ Update history at each epoch
        """
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


    def fit(self,
            train_ds=None,
            val_ds=None,
            epochs=0,
            initial_epoch=0,
            epoch_gen=0,
            epoch_disc=0,
            k=1,
            l1_lambda=100):
        """ Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            train_ds (tf.DataSet):
                    Input data as a dataset of tensors batchs.
            val_ds (tf.Dataset):
                    Data on which to evaluate the loss and any model metrics at
                    the end of each epoch. The model will not be trained on this data.
                    Note that the validation loss is affected by regularization layers
                    like noise and dropout.
            epochs (int):
                    Number of epochs to train the model. An epoch is an iteration
                    over the entire data provided. Note that in conjunction
                    with initial_epoch, epochs is to be understood as "final epoch".
                    The model is not trained for a number of iterations given by epochs,
                    but merely until the epoch of index epochs is reached.
            initial_epoch (int, optional):
                    Epoch at which to start training (useful for resuming a
                    previous training run). Defaults to 0
            epoch_gen (int, optional):
                    Number of epochs at which the generator is training alone at start,
                    driven only by its L1 loss. Note that, if epoch_gen = epochs,
                    then only the generator will train. Defaults to 0.
            epoch_disc (int, optional):
                    Number of epochs at which the discriminator is training alone,
                    right after the generator trained alone (if epoch_gen != 0)
                    Note that, if epoch_disc = epochs and epochs_gen = 0,
                    then only the generator will train. Defaults to 0.
            k (int, optional):
                    Number of epochs the discriminator trains before training
                    the generator for one epoch. Defaults to 1.
            l1_lambda (int, optional):
                    Weight on L1 term in objective. Defaults to 100.
        """

        # ==== INITIALIZING ====
        start_training = time.time()
        self.fit_history(epochs, epoch_gen, epoch_disc, l1_lambda)
        # Tracker used to follow up k loops for disc followed by 1 loop for generator
        k_tracker = 0
        # Define the trackers to track loss ..
        loss_tracker_train_gen = BinaryCrossentropy(
            name='loss_tracker_train_gen', from_logits=True)
        loss_tracker_train_disc = BinaryCrossentropy(
            name='loss_tracker_train_disc', from_logits=True)
        loss_tracker_val_gen = BinaryCrossentropy(
            name='loss_tracker_val_gen', from_logits=True)
        loss_tracker_val_disc = BinaryCrossentropy(
            name='loss_tracker_val_disc', from_logits=True)
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
        for epoch in range(initial_epoch, epochs):
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
                        metric_tracker_train_gen, l1_lambda)
                    # for epoch >= epoch_gen, train generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.train_discriminator_step(paint_train_batch,
                                                  image_train_batch,
                                                  loss_tracker_train_gen,
                                                  loss_tracker_train_disc,
                                                  metric_tracker_train_gen,
                                                  metric_tracker_train_disc)
                    # else, train generator + discriminator
                else:
                        # First train discriminator for k-1 steps
                    if k_tracker < k - 1:
                        self.train_discriminator_step(paint_train_batch,
                                                  image_train_batch,
                                                  loss_tracker_train_gen,
                                                  loss_tracker_train_disc,
                                                  metric_tracker_train_gen,
                                                  metric_tracker_train_disc)
                        k_tracker += 1
                        # then train generator+discriminator for 1 step
                    else:
                        k_tracker = 0
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
                                            metric_tracker_val_gen)
                # for epoch >= epoch_gen, validate generator + discriminator
                elif epoch < epoch_disc + epoch_gen:
                    self.val_discriminator_step(paint_val_batch,
                                                image_val_batch,
                                                loss_tracker_val_gen,
                                                loss_tracker_val_disc,
                                                metric_tracker_val_gen,
                                                metric_tracker_val_disc)
                    # else, validate generator + discriminator
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
            res_trackers_dict = dict(zip(trackers_name_list, res_trackers_list))
                # If epoch < epoch_gen, generator's loss is MAE and not binary-crossentropy.
                # Plus, disc loss/acc are set to -1 for plotting reasons
            if epoch < epoch_gen:
                res_trackers_dict['loss_tracker_train_gen'] = -1
                res_trackers_dict['loss_tracker_val_gen'] = -1
                res_trackers_dict['loss_tracker_train_disc'] = -1
                res_trackers_dict['loss_tracker_val_disc'] = -1
                res_trackers_dict['metric_tracker_train_disc'] = -1
                res_trackers_dict['metric_tracker_val_disc'] = -1
                
            self.update_history(start_training, start_epoch, epoch, res_trackers_dict)
            trackers_to_display = display_trackers(
                start_training, start_epoch, epoch, epoch_gen, epoch_disc,
                epochs, res_trackers_dict)
            generate_and_save_dashboard(self, epoch, train_ds, val_ds, trackers_to_display)
        
        # Generate one last display by plotting every epochs
        generate_and_save_dashboard(self, epoch,
                                 train_ds, val_ds, trackers_to_display,
                                 epochs_to_display=epochs)



if __name__ == "__main__":
    train, val, test = get_dataset(host='local',
                                   dataset='facades',
                                   batch_size=32)
    generator = make_dummy_generator()
    discriminator = make_dummy_discriminator(cgan_mode=True)
    cgan = CGAN(generator, discriminator, cgan_mode=True, random_sample=False)

    cgan.fit(train_ds=train,
               val_ds=val,
               epochs=10, epoch_gen=1, epoch_disc=1,
               l1_lambda=100)
