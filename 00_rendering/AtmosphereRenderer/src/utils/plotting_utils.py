from matplotlib import pyplot as plt
import numpy as np

def save_image_with_cmap(data, filename, cmap='inferno', dpi=300):
    vmin, vmax = np.min(data), np.max(data)
    normalized = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
    cmap = plt.cm.get_cmap(cmap)
    rgba = cmap(normalized)
    img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    img.save(filename, dpi=(dpi, dpi), format='PNG')

def plot_photometry(gray_array, title='Photometry', cmap='inferno'):
    plt.figure(dpi=300)
    plt.imshow(gray_array, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_spectral_map(specmap, title='Spectral Coverage Map'):
    plt.figure(dpi=300)
    plt.imshow(specmap)
    plt.title(title)
    plt.colorbar()
    plt.show()