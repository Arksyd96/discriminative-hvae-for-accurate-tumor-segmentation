import numpy as np
from scipy.stats import entropy

def make_grid(arr, samples=None, layout='col', rotate=False, idx=None):
    assert layout in ['col', 'row']
    assert idx.__len__() == samples if idx is not None else True

    if samples == None:
        samples = arr.shape[0]
    if idx is not None:
        arr = arr[idx]

    # rotate images with 45 degrees
    if rotate:
        arr = arr.transpose(0, 1, 3, 2)
        arr = arr[:, :, ::-1, :]
    
    f = np.vstack if layout == 'col' else np.hstack
    arr = f([f([arr[i][j] for j in range(arr.shape[1])]) for i in range(arr.shape[0])])
    return arr


def compute_divergence(real_masks, generated_masks):
    # Flatten the masks into 2D arrays
    real_masks_flat = real_masks.reshape(real_masks.shape[0], -1)
    generated_masks_flat = generated_masks.reshape(generated_masks.shape[0], -1)

    # Compute the probability distributions of the masks
    p_real = np.apply_along_axis(lambda x: np.bincount(x.astype(np.int64), minlength=2) / len(x), axis=1, arr=real_masks_flat)
    p_generated = np.apply_along_axis(lambda x: np.bincount(x.astype(np.int64), minlength=2) / len(x), axis=1, arr=generated_masks_flat)

    # Add a small epsilon to the probabilities to avoid division by zero
    epsilon = 1e-10
    p_real = np.clip(p_real, epsilon, 1.0)
    p_generated = np.clip(p_generated, epsilon, 1.0)

    # # Compute the average probability distribution between real and generated masks
    p_avg = 0.5 * (p_real + p_generated)

    # Add a small epsilon to the average probabilities to avoid division by zero
    p_avg = np.clip(p_avg, epsilon, 1.0)

    # Compute the KL divergence between the average distribution and the real and generated distributions
    kl_real_avg = entropy(p_real.T, p_avg.T)
    kl_generated_avg = entropy(p_generated.T, p_avg.T)

    # Compute the JSD and KLD
    jsd = 0.5 * (kl_real_avg + kl_generated_avg)
    kld = entropy(p_real.T, p_generated.T)

    return jsd, kld     


def inset_zoom_plot(images, bbox, label=''):
    plt.figure(figsize=(10, 20))
    for idx in range(images.shape[0]):
        ax = plt.subplot(1, 4, idx + 1)
        ax.imshow(np.rot90(images[idx]), cmap='gray')
        plt.axis('off')

        # Add a colored rectangle around the area of interest
        x, y = bbox[idx]
        rect = plt.Rectangle((x, y), 20, 20, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add a zoomed-in image of the area of interest
        axins = ax.inset_axes([0.54, 0.59, 0.5, 0.4])
        axins.imshow(np.rot90(images[idx])[y:y + 20, x:x + 20], cmap='gray')
        axins.set_xticks([])
        axins.set_yticks([])
        axins.spines['top'].set_color('r')
        axins.spines['right'].set_color('r')
        axins.spines['bottom'].set_color('r')
        axins.spines['left'].set_color('r')
        axins.spines['top'].set_linewidth(2)
        axins.spines['right'].set_linewidth(2)
        axins.spines['bottom'].set_linewidth(2)
        axins.spines['left'].set_linewidth(2)
    plt.show()