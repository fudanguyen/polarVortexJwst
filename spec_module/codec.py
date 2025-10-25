import h5py as h5 
import ast
import numpy as np
import matplotlib.pyplot as plt
import json

def readPhotometry(targetPath):
    """Read photometry data"""
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

path = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/rendering/atm_renderer/output/test_discrete.h5'

results = readPhotometry(path)


### Data structure of results
# for inclin, data in results.items():
#     f.create_dataset(f'{inclin}/gray_array', 
#                      data=np.array(data['gray_array']), 
#                      chunks=True, compression=compression)
#     f.create_dataset(f'{inclin}/specmask', data=data['specmask'])
#     f.create_dataset(f'{inclin}/metadata', data=str(data['metadata']))
#     f.create_dataset(f'{inclin}/time_array', data=data['time_array'])

