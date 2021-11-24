from matplotlib import pyplot as plt, ticker
from matplotlib import cycler

PLOT_SIZE_X = 10
PLOT_SIZE_Y = 5
PLOT_LEFT_POS = 0.1
PLOT_RIGHT_POS = 0.9
PLOT_BOTTOM_POS = 0.15
PLOT_TOP_POS = 0.85
PLOT_MARGIN = 0.01
PLOT_LW = 0.9
PLOT_GRID_LW = 0.2
PLOT_SAVE = False


def plot_loss(arrs, epochs, batch_size, learning_rate, optimizer, mode='loss'):
    colors = cycler('color',
                    # ['orange', 'dodgerblue'])
                    ['gold',
                     'darkorange',
                     'firebrick',
                     'y',
                     'forestgreen',
                     'turquoise',
                     'dodgerblue',
                     'rebeccapurple'])

    plt.rc('axes', prop_cycle=colors)

    fig, ax = plt.subplots(figsize=(PLOT_SIZE_X, PLOT_SIZE_Y))
    plt.subplots_adjust(left=PLOT_LEFT_POS, right=PLOT_RIGHT_POS, bottom=PLOT_BOTTOM_POS, top=PLOT_TOP_POS)
    plt.margins(x=PLOT_MARGIN)

    for phase in ['train', 'test']:
        plt.plot(range(0, epochs + 1), arrs[phase], label=phase, linewidth=PLOT_LW)

    plt.title(f"Train and validation {mode} (optimizer={optimizer}, learning rate={learning_rate}, batch={batch_size})")
    plt.xlabel("epoch")
    plt.ylabel(mode)

    legend = plt.legend(loc='best')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('white')

    plt.grid(axis='y', lw=PLOT_GRID_LW)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(epochs / 10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(epochs / 100))

    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=epochs)

    if mode == 'loss':
        ax.set_ylim(ymin=0)
        y_ticks_interval = 0.1
    else:
        ax.set_ylim(ymax=1)
        y_ticks_interval = 0.05

    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_ticks_interval))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_ticks_interval / 5))

    if PLOT_SAVE:
        plt.savefig(f'./graph_{mode}.png')
    plt.show()
