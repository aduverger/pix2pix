import random
import time
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from tensorflow import expand_dims


def display_trackers(start_training, start_epoch, epoch, epoch_gen, epoch_disc, epochs, res_trackers_dict):
    if epoch < epoch_gen:
        display_str = f"\n Training Phase - Generator ({epoch+1}/{epoch_gen})\n"
    elif epoch < epoch_gen + epoch_disc:
        display_str = f"\n Training Phase - Discriminator ({epoch+1-epoch_gen}/{epoch_disc})\n"
    else:
        display_str = f"\n Training Phase : GAN ({epoch+1-epoch_gen-epoch_disc}/{epochs-epoch_disc-epoch_gen})\n"

    display_str += f'''
        Epoch {epoch+1:3}/{epochs:3}
        Elapsed time
                    - since training        {str(datetime.timedelta(seconds=round(time.time()-start_training, 0)))}
                    - since last epoch    {str(datetime.timedelta(seconds=round(time.time()-start_epoch, 0)))}
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
                         Discriminator loss = {res_trackers_dict['loss_tracker_train_disc']:0.2f}          Discriminator accuracy = {res_trackers_dict['metric_tracker_train_disc']:0.2f}

        Val set :   Generator GAN loss = {res_trackers_dict['loss_tracker_val_gen']:0.2f}        Generator MAE = {res_trackers_dict['metric_tracker_val_gen']:0.2f}
                         Discriminator loss = {res_trackers_dict['loss_tracker_val_disc']:0.2f}          Discriminator accuracy = {res_trackers_dict['metric_tracker_val_disc']:0.2f}
        '''
    return display_str


def display_image(ax, sample_tensor):
    ax.imshow((sample_tensor * 127.5 + 127.5).numpy().astype('uint8'))
    ax.axis('off')


def save_image(sample_tensor, epoch):
    plt.imshow((sample_tensor * 127.5 + 127.5).numpy().astype('uint8'))
    plt.axis('off')
    plt.savefig('output_at_epoch_{:04d}.png'.format(epoch))


def generate_and_save_dashboard(cgan, epoch, train_ds, val_ds,
                             trackers_to_display, epochs_to_display=50,
                             display_trackers=True, display_plots=True):
    display.clear_output(wait=True)
    #TODO: Use next_iter to avoid iterating upon the whole datasets ?
    train_list = [(paint, real) for paint, real in iter(train_ds)]
    val_list = [(paint, real) for paint, real in iter(val_ds)]

    if cgan.random_sample:
        index_batch_train = random.randint(0, len(train_list) - 1)
        index_batch_val = random.randint(0, len(val_list) - 1)
        index_train = random.randint(0, train_list[index_batch_train][0].shape[0] - 1)
        index_val = random.randint(0, val_list[index_batch_val][0].shape[0] - 1)
        cgan.paint_train = train_list[index_batch_train][0][index_train]
        cgan.paint_val = val_list[index_batch_val][0][index_val]
        cgan.real_train = train_list[index_batch_train][1][index_train]
        cgan.real_val = val_list[index_batch_val][1][index_val]

    prediction_train = cgan.generator(expand_dims(
                            cgan.paint_train, axis=0),
                                      training=False)
    prediction_val = cgan.generator(expand_dims(
                            cgan.paint_val, axis=0),
                                    training=False)

    #save_image(prediction_val[0], epoch)
    
    fig = plt.figure(constrained_layout=True, figsize=(18,10))
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

    display_image(ax1, cgan.paint_train)
    ax1.set_title(label="Train sample \n Input")
    display_image(ax2, prediction_train[0])
    ax2.set_title(label=f"Output")
    display_image(ax3, cgan.real_train)
    ax3.set_title(label="Ground truth")
    display_image(ax4, cgan.paint_val)
    ax4.set_title(label="Val sample \n Input")
    display_image(ax5, prediction_val[0])
    ax5.set_title(label="Output")
    display_image(ax6, cgan.real_val)
    ax6.set_title(label="Ground truth")
    
    if display_trackers:
        ax7.text(0, 1, trackers_to_display, ha='left', size='medium')
        ax7.axis('off')
        
    if display_plots:
        plot_last_n_epochs(ax8, cgan.history, n=epochs_to_display, set_name='train', show_label=False)
        plot_last_n_epochs(ax9, cgan.history, n=epochs_to_display, set_name='val', show_label=True)
    
    fig.savefig('display_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def plot_last_n_epochs(ax=None, history=None, n=50, set_name='train', show_label=True):
    twin = ax.twinx()
    l1_lambda = history['l1_lambda'][-1]
    scaled_mae_list = [mae * l1_lambda for mae in history[set_name]['gen_mae']]
    epoch_gen = history['epoch_gen'][-1]
    epoch_disc = history['epoch_disc'][-1]
    last_epoch = max(n, history['epoch_index'][-1])
    start_epoch = max(0, last_epoch - n)
    loss_max = max(
        1,
        max(history['train']['gen_loss'][start_epoch:]),
        max(history['train']['disc_loss'][start_epoch:]),
        max(history['val']['gen_loss'][start_epoch:]),
        max(history['val']['disc_loss'][start_epoch:])
        )
    mae_max = max(
        max([mae * l1_lambda for mae in history['train']['gen_mae'][start_epoch:]]),
        max([mae * l1_lambda for mae in history['val']['gen_mae'][start_epoch:]])
        )
    sns.lineplot(x=history['epoch_index'],
                 y=history[set_name]['gen_loss'],
                 ax=ax, color='tab:blue', label='Generator GAN loss')
    sns.lineplot(x=history['epoch_index'],
                 y=history[set_name]['disc_loss'],
                 ax=ax, color='tab:red', label='Discriminator loss')
    sns.lineplot(x=history['epoch_index'],
                 y=scaled_mae_list,
                 ax=twin, color='tab:cyan', label='Generator MAE * λ')
    twin.lines[0].set_linestyle("--")

    # GENERATOR TRAINING AREA
    if epoch_gen != 0 and last_epoch > epoch_gen > start_epoch:
        if not show_label:
            ax.text((epoch_gen+start_epoch)/2, -0.2, 'Generator\ntraining', color='tab:blue', ha='center', va='top')
        ax.fill_between(np.arange(0, epoch_gen, 0.01), 0, 1,
                    color='tab:blue', alpha=0.05, transform=ax.get_xaxis_transform())

    # DISCRIMINATOR TRAINING AREA
    if epoch_disc != 0 and last_epoch > epoch_disc + epoch_gen > start_epoch:
        if not show_label:
            ax.text(epoch_disc/2 + epoch_gen, -0.2, 'Discriminator\ntraining', color='tab:red', ha='center', va='top')
        ax.fill_between(np.arange(epoch_gen, epoch_gen+epoch_disc, 0.01), 0, 1,
                    color='tab:red', alpha=0.05, transform=ax.get_xaxis_transform())

    # LEGEND
    if show_label:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines2 + lines, labels2 + labels, loc=2)
        twin.legend([],[], frameon=False)
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    else:
        ax.legend([],[], frameon=False)
        twin.legend([],[], frameon=False)

    # X-Y LABELS
    if show_label:
        ax.set_xlabel('N epochs')
    ax.set_ylabel('Loss')
    twin.set_ylabel(ylabel='MAE')
    ax.set(xlim=(start_epoch, last_epoch), ylim=(0, loss_max))
    twin.set(xlim=(start_epoch, last_epoch), ylim=(0, mae_max))
    ax.tick_params(left=False, bottom=False)
    twin.tick_params(right=False)

    # TITLE
    ax.set_title(f'{set_name.capitalize()} set',
                  fontdict={
                      'fontsize': 13,
                      'fontweight': 7,
                      'color': 'black'
                  })
    sns.despine(left=True, bottom=True)
    ax.grid(axis='both', ls='--', alpha=0.5)    
    
    return ax, twin


if __name__ == "__main__":
    history = {
            'epochs': 50,
            'epoch_gen': 20,
            'epoch_disc': 10,
            'l1_lambda': 100,
            'epoch_index': range(0,50),
            'time_epoch': [],
            'time_cumulative': [],
            'train': {
                'gen_loss': np.linspace(0.1, 0.8, 50),
                'disc_loss': np.linspace(0.2, 0.7, 50),
                'gen_mae': np.linspace(20, 30, 50),
                'disc_acc': [],
            },
            'val': {
                'gen_loss': [],
                'disc_loss': [],
                'gen_mae': [],
                'disc_acc': [],
            }
        }
    fig, ax = plt.sublots(1,1)
    ax, twin = plot_last_n_epochs(ax, history, n=50)
    plt.show()