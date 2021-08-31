import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_last_n_epochs(ax, history, n=50, set_name='train', show_label=True):
    twin = ax.twinx()
    l1_lambda = history['l1_lambda']
    scaled_mae_list = [mae * l1_lambda for mae in history[set_name]['gen_mae']]
    epoch_gen = history['epoch_gen']
    epoch_disc = history['epoch_disc']
    last_epoch = history['epoch_index'][-1]
    start_epoch = max(0, last_epoch - n)
    loss_max = max(
        max(history['train']['gen_loss']),
        max(history['train']['disc_loss']),
        max(history['val']['gen_loss']),
        max(history['val']['disc_loss'])
    )
    mae_max = max(
        max(history['train']['gen_mae']),
        max(history['val']['gen_mae'])
    )
    sns.lineplot(x=history['epoch_index'], y=history[set_name]['gen_loss'], ax=ax, color='tab:blue', label='Generator loss')
    sns.lineplot(x=history['epoch_index'], y=history[set_name]['disc_loss'], ax=ax, color='tab:red', label='Discriminator loss')
    sns.lineplot(x=history['epoch_index'], y=scaled_mae_list, ax=twin, color='tab:cyan', label='Generator MAE * λ')
    twin.lines[0].set_linestyle("--")

    # GENERATOR TRAINING AREA
    if epoch_gen != 0 and epoch_gen > start_epoch:
        ax.text((epoch_gen+start_epoch)/2, -0.05, 'Generator\ntraining', color='tab:blue', ha='center', va='top')
        ax.fill_between(np.arange(0, epoch_gen, 0.01), 0, 1,
                    color='tab:blue', alpha=0.05, transform=ax.get_xaxis_transform())

    # DISCRIMINATOR TRAINING AREA
    if epoch_disc != 0 and epoch_disc > last_epoch:
        ax.text(epoch_disc/2 + epoch_gen, -0.05, 'Discriminator\ntraining', color='tab:red', ha='center', va='top')
        ax.fill_between(np.arange(epoch_gen, epoch_gen+epoch_disc, 0.01), 0, 1,
                    color='tab:red', alpha=0.05, transform=ax.get_xaxis_transform())

    # LEGEND
    if show_label:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines2 + lines, labels2 + labels, loc=2)
        twin.legend([],[], frameon=False)
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.2), ncol=3, title=None, frameon=False)
    else:
        ax.legend([],[], frameon=False)
        twin.legend([],[], frameon=False)

    # X-Y LABELS
    ax.set_xlabel('N epochs', loc='right')
    ax.set_ylabel('Loss', loc='top')
    twin.set_ylabel(ylabel='MAE', loc='top')
    ax.set(xlim=(start_epoch, max(n, last_epoch)), ylim=(0, loss_max))
    twin.set(xlim=(start_epoch, max(n, last_epoch)), ylim=(0, mae_max))
    ax.tick_params(left=False, bottom=False)
    twin.tick_params(right=False)

    # TITLE
    ax.set_title(f'{set_name} set')
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
    ax, twin = plot_last_n_epochs(history, 50)
    plt.show()