import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os


def draw(data, map='gray', caxis=None):
    """Draw an image"""
    plt.ion()
    create_figure(data, map, caxis)
    plt.show()
    plt.pause(0.01)


def plot(data):
    """plot a graph"""
    plt.plot(data)
    plt.show()


def save_draw(data, storage_directory, file_name, map='gray', caxis=None):
    """save an image"""
    create_figure(data, map, caxis)

    full_path = get_full_path(storage_directory, file_name)
    plt.savefig(full_path)
    plt.close()


def save_plot(data, storage_directory, file_name):
    """save a graph"""
    full_path = get_full_path(storage_directory, file_name)
    plt.plot(data)
    plt.savefig(full_path)
    plt.close()


def save_numpy_array(data, storage_directory, file_name):
    """save a numpy array in .npy format"""

    full_path = get_full_path(storage_directory, file_name)

    np.save(full_path, data)


def load_numpy_array(storage_directory, file_name):
    """load a .npy file into numpy array"""

    full_path = os.path.join(storage_directory, file_name)

    # add .npy extension if needed
    if not full_path.endswith('.npy'):
        full_path = full_path + '.npy'

    if not os.path.exists(full_path):
        raise Exception('File named ' + full_path + ' does not exist')

    return np.load(full_path)


def get_full_path(storage_directory, file_name):
    # create storage_directory if needed
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    full_path = os.path.join(storage_directory, file_name)

    return full_path


def create_figure(data, map, caxis=None):
    fig, ax = plt.subplots()

    plt.axis('off')  # no axes

    if caxis is None:
        im = plt.imshow(data, cmap=map)
    else:
        im = plt.imshow(data, cmap=map, vmin=caxis[0], vmax=caxis[1])

    # equal aspect ratio
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    # add colorbar
    plt.colorbar(im, orientation='vertical')


def multiple_figures(*args, map, caxis=None):
    fig, axes = plt.subplots(1, len(args))
    for index, ax in enumerate(axes):
        if caxis is None:
            im = ax.imshow(args[index], cmap=map, aspect="auto")

        else:
            im = ax.imshow(args[index], cmap=map, vmin=caxis[0], vmax=caxis[1], aspect="auto")

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


def multiple_figures_bar(*args, map, caxis=None):
    fig, axes = plt.subplots(1, len(args))
    for index, ax in enumerate(axes):
        if caxis is None:
            im = ax.imshow(args[index], cmap=map, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(args[index], cmap=map, vmin=caxis[0], vmax=caxis[1], aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def multiple_figures_2(data, map, caxis=None, titles=None):
    fig, axes = plt.subplots(1, len(data))
    test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    if titles is None:
        for index, ax in enumerate(axes):
            if caxis is None:
                im = ax.imshow(data[index], cmap=map, aspect="auto")
                ax.set_title('({})'.format(test_list[index]), y=-0.1)
                ax.axis('off')

            else:
                im = ax.imshow(data[index], cmap=map, vmin=caxis[0], vmax=caxis[1], aspect="auto")
                ax.set_title('({})'.format(test_list[index]), y=-0.1)
                ax.axis('off')
    else:
        for index, ax in enumerate(axes):
            if caxis is None:
                im = ax.imshow(data[index], cmap=map, aspect="auto")
                ax.set_title('({}) {} angles'.format(test_list[index], titles[index]), y=-0.15, fontsize=14)
                ax.axis('off')

            else:
                im = ax.imshow(data[index], cmap=map, vmin=caxis[0], vmax=caxis[1], aspect="auto")
                ax.set_title('({}) {} angles'.format(test_list[index], titles[index]), y=-0.15, fontsize=14)
                ax.axis('off')

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


def multiple_figures_bar_2(data, map, caxis=None):
    fig, axes = plt.subplots(1, len(data))
    for index, ax in enumerate(axes):
        if caxis is None:
            im = ax.imshow(data[index], cmap=map, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        else:
            im = ax.imshow(data[index], cmap=map, vmin=caxis[0], vmax=caxis[1], aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
