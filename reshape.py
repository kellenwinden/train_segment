from shutil import copy
import h5py
import json

def reshape(model_path):
    new_window = (None, 1024, 1024)
    path = model_path[0:-5] + "_1024.hdf5"
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
    reshape('unet_calcium.hdf5')
