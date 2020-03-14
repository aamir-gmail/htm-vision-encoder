from math import floor
import os.path
from os import path
import keras
import numpy as np
from htm.bindings.sdr import SDR, Metrics
import pickle
from htm.bindings.algorithms import SpatialPooler
from keras import Input, Model, regularizers
from keras.engine.saving import model_from_json
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape, Dense, BatchNormalization, \
    ZeroPadding2D
from sklearn.manifold import TSNE
import htm.bindings.encoders
from keras.datasets import mnist
import scipy.misc
from PIL import Image
from htm.encoders.rdse import RDSE, RDSE_Parameters
from numba import vectorize, float32, float64

ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
import PIL

SCALE_FACTOR = 2
MATCH_FACTOR = 168


#  Auto encoder CNN for encoding MNISTdata set to 128 Dimensions embeddings
def auto_encs():
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    x = ZeroPadding2D((1, 1))(input_img)
    x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Flatten()(encoded)
    encoded = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dense(128, activation='selu')(encoded)  # Cut of the network here after training.
    encoded = Reshape((4, 4, 8))(encoded)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(64, (3, 3), activation='selu', padding='same')(encoded)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='selu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    cnn_embeddings = Model(input_img, decoded)
    cnn_embeddings.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # try as well mse
    print(cnn_embeddings.summary())
    return cnn_embeddings


# train a split the model here.
def pre_data(model=None):
    if model is None:
        return 0
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / (255 - 1)
    x_test = x_test.astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    model.fit(x_train, x_train,
              epochs=20,
              batch_size=64,
              shuffle=True,
              validation_data=(x_test, x_test))
    # After we have trained the auto encoder we,
    # split the decoder part and keep the encoder
    # the 128 D output of the encoder will be used to
    # create input to RDSE via TSNE
    model_new = Model(input=model.layers[0].input,
                      output=model.layers[15].output)
    return model_new


# Save model to disk
def save_model(model):
    model_json = model.to_json()
    with open('model/embedding_v3.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/embedding_v3.h5")
    print("Saved model to disk")


# load model from disk
def load_model():
    json_file = open('model/embedding_v3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/embedding_v3.h5")
    print("Loaded model from disk")
    return loaded_model


# Yet to be used K-winner
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# Scale vector to be used in conjunction with SDR resolution parameter.
@vectorize(['float32(float32)', 'float64(float64)'], target='parallel')
def scale_vector(x):
    return x * SCALE_FACTOR


# Get the output from auto encoder and perp for SDR generation
def predict_and_reduce(new_model=None):
    pred = []
    norms = []
    if new_model is None:
        return 0
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    anm = np.asarray(PIL.Image.open('images/anm.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm1 = np.asarray(PIL.Image.open('images/anm1.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm2 = np.asarray(PIL.Image.open('images/anm2.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    x_train = x_train.astype('float32') / (255 - 1)
    x_test = x_test.astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.concatenate((x_test, anm, anm1, anm2))
    predections = new_model.predict(x_test)
    for vec in predections:
        norms.append(np.linalg.norm(vec))
    # Used TSNE to reduce dimensions for data prep for  RDSE encoder
    tsne = TSNE(n_components=3)
    X_hat = tsne.fit_transform(predections)
    X_hat = scale_vector(X_hat)
    # Get the norm of the vector (128) as input to RDSE
    # Adding this parameter helps SP get better results
    norms = np.array(norms).reshape(-1, 1)
    # Serialise the output for the next stage.
    X_hat = np.concatenate([X_hat, norms], axis=1)
    pickle.dump(X_hat, open('data/x_hat_v3.pkl', mode='wb'))


# Setup the RDSE encoder for imput SDR
# in this method single RDSE is used to encode all input
def sacler_data_randonscaler_method_1():
    pooler_data = []
    data = pickle.load(open('data/x_hat_v3.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    col4 = data[:, 3:4].flatten()
    parameter1 = RDSE_Parameters()
    parameter1.size = 2000
    parameter1.sparsity = 0.02
    parameter1.resolution = 0.66
    rsc1 = RDSE(parameter1)
    # Create SDR for 3D TSNE input plus one for magnitude for 128 D original vector.
    # Loop through all vectors one at the time
    # to create SDe for SP.
    for _x1, _x2, _x3, _x4 in zip(col1, col2, col3, col4):
        x_x1 = rsc1.encode(_x1)
        x_x2 = rsc1.encode(_x2)
        x_x3 = rsc1.encode(_x3)
        x_x4 = rsc1.encode(_x4)
        pooler_data.append(SDR(8000).concatenate([x_x1, x_x2, x_x3, x_x4]))
    return pooler_data

# Setup the RDSE encoder for imput SDR
# In this method we use diffreent RDSE for encoding all vector as SDR
def sacler_data_randonscaler_method_2():
    pooler_data = []
    data = pickle.load(open('data/x_hat_v3.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    col4 = data[:, 3:4].flatten()
    parameter1 = RDSE_Parameters()
    parameter1.size = 2000
    parameter1.sparsity = 0.02
    parameter1.resolution = 0.66
    rsc1 = RDSE(parameter1)
    parameter2 = RDSE_Parameters()
    parameter2.size = 2000
    parameter2.sparsity = 0.02
    parameter2.resolution = 0.66
    rsc2 = RDSE(parameter2)
    parameter3 = RDSE_Parameters()
    parameter3.size = 2000
    parameter3.sparsity = 0.02
    parameter3.resolution = 0.66
    rsc3 = RDSE(parameter3)
    parameter4 = RDSE_Parameters()
    parameter4.size = 2000
    parameter4.sparsity = 0.02
    parameter4.resolution = 0.66
    rsc4 = RDSE(parameter4)
    # Create SDR for 3D TSNE input plus one for magnitude for 128 D original vector.
    for _x1, _x2, _x3, _x4 in zip(col1, col2, col3, col4):
        x_x1 = rsc1.encode(_x1)
        x_x2 = rsc2.encode(_x2)
        x_x3 = rsc3.encode(_x3)
        x_x4 = rsc4.encode(_x4)
        pooler_data.append(SDR(8000).concatenate([x_x1, x_x2, x_x3, x_x4]))
    return pooler_data


# Create SP
def spatial_pooler_encoer(pooler_data):
    sp1 = SpatialPooler(
        inputDimensions=(8000,),
        columnDimensions=(8000,),
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=0.0335,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13999999999999999,
        boostStrength=4.0,
        wrapAround=True
    )
    sdr_array = []
    # We run SP over there epochs and in the third epoch collect the results
    # this technique yield betters results than  a single epoch
    for encoding in pooler_data:
        activeColumns1 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns1)
    for encoding in pooler_data:
        activeColumns2 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns2)
    for encoding in pooler_data:
        activeColumns3 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns3)
        sdr_array.append(activeColumns3)
    # To make sure the we can relate SP output to real images
    # we take out specific SDR which corrospond to known images.
    hold_out = sdr_array[10000]
    hold_out1 = sdr_array[10001]
    hold_out2 = sdr_array[10002]
    (x_train, _), (x_test, _) = mnist.load_data()
    anm = np.asarray(PIL.Image.open('images/anm.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm1 = np.asarray(PIL.Image.open('images/anm1.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm2 = np.asarray(PIL.Image.open('images/anm2.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    x_train = x_train.astype('float32') / (255 - 1)
    x_test = x_test.astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.concatenate((x_test, anm, anm1, anm2))
    counter = 0
    # finally we loop over the SP SDR and related image to get the once
    # which have a greater overlap with the image we are searching for
    for _s, _img in zip(sdr_array, x_test):
        _x_ = hold_out2.getOverlap(_s)
        if _x_ > 138 : # Adjust as required.
            _img_ = _img.reshape((28, 28))
            _img_ = (_img_ * 254).astype(np.uint8)
            im = Image.fromarray(_img_).convert('RGB')
            im.save('test_results/' + str(counter) + 'outfile.jpg')
            print('Sparsity - ' + str(_s.getSparsity()))
            print(_x_)
            print(str('counter - ') + str(counter))
            counter += 1
            # Write all images to file which have good overlap with the target  image


# All code is self contained wih no external dependencies.
def main():
    # model = auto_encs()
    # new_model = pre_data(model)
    # save_model(new_model)
    # Re-run the part if you incorporate new test images.
    # new_model = load_model()
    # predict_and_reduce(new_model)
    pooler_data = sacler_data_randonscaler_method_2()
    spatial_pooler_encoer(pooler_data)
    strs = ' '


if __name__ == '__main__':
    main()
