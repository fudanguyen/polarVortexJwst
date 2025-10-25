import h5py as h5 
import ast
import numpy as np
import matplotlib.pyplot as plt

def readPhotometry(targetPath):
    results = {}
    with h5.File(path, 'r') as f:
        for inclin, data in f.items():
            metadata_bytes = data['metadata'][()]
            metadata_str = metadata_bytes.decode('utf-8')

            # Replace TimeConfig(...) with a dict
            # remove this in metadata handling. in main.py
            metadata_str = metadata_str.replace(
                "TimeConfig(t0=0, t1=60, frames=120)",
                "{'t0': 0, 't1': 60, 'frames': 120}"
            )

            metadata_dict = ast.literal_eval(metadata_str)

            results[inclin] = {
                'gray_array': data['gray_array'][:],
                'specmask': data['specmask'][:],
                'metadata': metadata_dict,
                'time_array': data['time_array'][:]
            }
    return results

path = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/rendering/atm_renderer/output/test_discrete.h5'

results = readPhotometry(path)

print( results['0']['gray_array'].shape )
print( results['0']['specmask'].shape )
print( results['0']['metadata']['band_config'] )

### Data structure of results
# for inclin, data in results.items():
#     f.create_dataset(f'{inclin}/gray_array', 
#                      data=np.array(data['gray_array']), 
#                      chunks=True, compression=compression)
#     f.create_dataset(f'{inclin}/specmask', data=data['specmask'])
#     f.create_dataset(f'{inclin}/metadata', data=str(data['metadata']))
#     f.create_dataset(f'{inclin}/time_array', data=data['time_array'])

