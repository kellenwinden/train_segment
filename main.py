import numpy as np
import model
import skimage.io as io
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def train_model(name):
    #Read in file prefixes
    with open(name) as f:
        file_prefixes = f.readlines()

    #Read data and masks
    img_data = [[] for i in range(len(file_prefixes))]
    for i, n in enumerate(file_prefixes):
        if n[-1] == '\n': n = n[0:-1]
        img_data[i] = [[] for j in range(3)]
        img = io.imread(n + ".tif")
        img_data[i][0] = img/np.max(img)
        f = open(n + "_masks.json")
        masks = json.load(f)
        f.close()
        img_data[i][1] = create_mask(masks, img_data[i][0].shape)
        img_data[i][2] = calculate_mask_center(masks)

    #Call generator function
    batch_gen = gen_func(16, (256, 256), img_data)

    #Create and compile model
    model_unet = model.unet()
    model_unet.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    #Fit model
    model_checkpoint = ModelCheckpoint('unet_calcium.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model_unet.fit(batch_gen, steps_per_epoch=200, epochs=10, callbacks=[model_checkpoint])

def gen_func(batch_size, window_size, img_data):
    augment_funcs = [
        lambda a: a,  # Identity.
        lambda a: a[:, ::-1],  # Horizontal flip.
        lambda a: a[::-1, :],  # Vertical flip.
        lambda a: np.rot90(a, 1),  # 90 deg rotations.
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a, 3),
    ]

    s_batch = np.zeros((batch_size, window_size[0], window_size[1]), dtype=np.float32)
    m_batch = np.zeros((batch_size, window_size[0], window_size[1]), dtype=np.uint8)

    total_batches = 0
    print_change_to_random = True
    while True:
        for i in range(batch_size):
            img_number = np.random.randint(len(img_data))
            img_size = img_data[img_number][0].shape
            neuron_number = np.random.randint(len(img_data[img_number][2]))

            if total_batches < 800:
                tile_center = img_data[img_number][2][neuron_number]
                tile_center[0] += np.random.randint(window_size[0]/2)
                tile_center[1] += np.random.randint(window_size[0]/2)
            else:
                if print_change_to_random:
                    print("Changing to random tiles")
                    print_change_to_random = False
                tile_center = [np.random.randint(img_size[0]), np.random.randint(img_size[1])]

            x0 = int(tile_center[0] - window_size[1]/2)
            y0 = int(tile_center[1] - window_size[0]/2)

            if x0 < 0: x0 = 0
            elif (x0 + window_size[1]) >= img_size[1]: x0 = img_size[1] - window_size[1]
            if y0 < 0: y0 = 0
            elif (y0 + window_size[0]) >= img_size[0]: y0 = img_size[0] - window_size[0]

            s_slice = img_data[img_number][0][x0:(x0+window_size[1]),y0:(y0+window_size[1])]
            m_slice = img_data[img_number][1][x0:(x0+window_size[1]),y0:(y0+window_size[1])]

            aug = augment_funcs[np.random.randint(6)]
            s_batch[i,:,:] = aug(s_slice)
            m_batch[i,:,:] = aug(m_slice)
        total_batches +=  1
        yield s_batch, m_batch

def create_mask(masks, img_size):
    mask = np.zeros((img_size))
    for m in range(len(masks)):
        x = np.array(masks[m]['coordinates']).T.tolist()
        mask[tuple([x[1],x[0]])] = 1
    return mask

def calculate_mask_center(masks):
    mask_center = [[] for i in range(len(masks))]
    for i in range(len(masks)):
        center = np.mean(masks[i]['coordinates'],axis=0)
        mask_center[i] = (int(center[0]), int(center[1]))
    return mask_center

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_model('/Users/kellenwinden/Data/calcium_imaging/train_segment/file_prefixes.txt')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
