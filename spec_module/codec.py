import h5py as h5 
import ast
import numpy as np
import matplotlib.pyplot as plt
import json

def readPhotometry(targetPath):
    """Read photometry data: gray_array, specmask, metadata, time_array.
    ### Data structure of results
        # for inclin, data in results.items():
        #     f.create_dataset(f'{inclin}/gray_array', 
        #                      data=np.array(data['gray_array']), 
        #                      chunks=True, compression=compression)
        #     f.create_dataset(f'{inclin}/specmask', data=data['specmask'])
        #     f.create_dataset(f'{inclin}/metadata', data=str(data['metadata']))
        #     f.create_dataset(f'{inclin}/time_array', data=data['time_array'])
        # return: results dict"""
    results = {}
    with h5.File(targetPath, 'r') as f:
        for inclin in f.keys():
            data = f[inclin]

            results[inclin] = {
                'gray_array': data['gray_array'][:],
                'specmask': data['specmask'][:],
                'metadata': json.loads(data['metadata'][()].decode('utf-8')),
                'time_array': data['time_array'][:]
            }
    for inclin in list(results.keys())[:1]:
        print("Each inclination contains:")
        print(" - gray_array: ", results[inclin]['gray_array'].shape)
        print(" - specmask: ", results[inclin]['specmask'].shape)
        for meta in results[inclin]['metadata'].keys():
            if meta == 'band_config':
                print("   - ", meta, ": ")
                for band in results[inclin]['metadata'][meta]:
                    print("     - ", band)
            else: print("   - ", meta, ": ", results[inclin]['metadata'][meta])
        print(" - time_array: ", results[inclin]['time_array'].shape)

    return results

def generate_bins(a, b, nbin, type='linear', power=2):
        """
        Generate bins with power-law spacing.
        """
        if type == 'linear':
            bins = np.linspace(a, b, nbin)
        elif type == 'power':
            # Generate interior points only (excluding 0 and 1)
            t = np.linspace(0, 1, nbin)  # Exclude endpoints
            normalized = t ** power
            bins = a + normalized * (b - a)
        return [0] + bins.tolist()

# Example usage
path = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/rendering/atm_renderer/output/test_discrete.h5'
results = readPhotometry(path)

# Get clim to variability pixel-value range
cl1, cl2 = results['40']['metadata']['colorlim']
Fpolar_var = results['40']['metadata']['Fpolar_var']
Fband_var = results['40']['metadata']['Fband_var']
var = max(Fpolar_var, Fband_var)
v1, v2 = 1 - var, 1 + var

# Calculate the linear coefficients for mapping [cl1, cl2] to [0, 255]
slope = 255/(cl2 - cl1)
intercept = 0 - slope * cl1
a, b = slope*v1 + intercept, slope*v2 + intercept

# Generate bins
bins = generate_bins(a, b, nbin=10, type='linear')
print(bins)

# Function to digitize frames into cloud-thickness codecs
def digitize_frames(results, bins):
    """Digitize all frames in gray_array for each inclination."""
    for inclin in results.keys():
        gray_array = results[inclin]['gray_array']
        digitized = np.digitize(gray_array, bins, right=True)  # Vectorized operation
        results[inclin]['digitized'] = digitized

digitize_frames(results, bins)
#%%
# Plot first 4 frames in a compact 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ite = 4
for i, ax in enumerate(axes.flat):
    im = results['40']['digitized'][i*ite]
    ax.imshow(im, cmap='viridis')
    ax.set_title(f"Frame {i*ite}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Unique pixel values
print(np.unique(im))

# %%
specmask = results['40']['specmask']
speckey = list(results['40']['metadata']['speckey'].keys())



# %%
