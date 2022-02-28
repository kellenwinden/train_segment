from shutil import copy
import h5py
import json

def reshape(model_path, size):
    new_window = (None, size, size)
    path = model_path[0:-5] + f"_{size}.hdf5"
    copy(model_path, path)
    h5 = h5py.File(path, 'a')
    config = json.loads(h5.attrs['model_config'])

    for layer in config['config']['layers']:
        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'] = new_window

        if 'output_shape' in layer['config']:
            layer['config']['output_shape'] = new_window

    h5.attrs['model_config'] = json.dumps(config).encode()
    h5.close()

if __name__ == '__main__':
    reshape('unet_calcium.hdf5', 512)
    reshape('unet_calcium.hdf5', 1024)
