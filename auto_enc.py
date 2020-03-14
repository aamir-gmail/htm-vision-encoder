from math import floor

import keras
import numpy as np
from htm.bindings.sdr import SDR, Metrics
import pickle
from htm.bindings.algorithms import SpatialPooler
from keras import Input, Model, regularizers
from keras.engine.saving import model_from_json
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape, Dense,BatchNormalization
from sklearn.manifold import TSNE
import htm.bindings.encoders
from keras.datasets import mnist
import scipy.misc
from PIL import Image
from htm.encoders.rdse import RDSE, RDSE_Parameters


ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
import PIL


def auto_encs():
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Flatten()(encoded)
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Reshape((4, 4, 8))(encoded)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(encoded)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    neurl_embedding = Model(input_img, decoded)
    neurl_embedding.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # try as well mse
    print(neurl_embedding.summary())
    return neurl_embedding


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
              batch_size=128,
              shuffle=True,
              validation_data=(x_test, x_test))
    model_new = Model(input=model.layers[0].input,
                      output=model.layers[10].output)
    return model_new


def save_model(model):
    model_json = model.to_json()
    with open("model/embedding.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/embedding.h5")
    print("Saved model to disk")


def load_model():
    json_file = open('model/embedding.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/embedding.h5")
    print("Loaded model from disk")
    return loaded_model


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def predict_and_reduce(new_model=None):
    pred =[]
    if new_model is None:
        return 0
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    anm = np.asarray(PIL.Image.open('images/anm.jpg').convert('L')).reshape((1,28,28,1)) / (255-1)
    anm1 =np.asarray(PIL.Image.open('images/anm1.jpg').convert('L')).reshape((1,28,28,1)) / (255-1)
    anm2 = np.asarray(PIL.Image.open('images/anm2.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    x_train = x_train.astype('float32') / (255 - 1)
    x_test = x_test.astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.concatenate((x_test,anm,anm1,anm2))
    predections = new_model.predict(x_test)
    for _y in predections:
        idx = largest_indices(_y, 14)
        idx = idx[0].tolist()
        for _y_ in range(0,128):
            if _y_ not in idx:
                _y[_y_] = 0
        pred.append(np.array(_y))
    pred1 = np.asarray(pred)
    tsne = TSNE(n_components=3)
    X_hat = tsne.fit_transform(pred1)
    pickle.dump(X_hat, open('data/x_hat.pkl', mode='wb'))

def sacler_data_randonscaler():
    pooler_data = []
    data = pickle.load(open('data/x_hat.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    parameter1 = RDSE_Parameters()
    parameter1.size = 2000
    parameter1.sparsity = 0.02
    parameter1.resolution = 0.88
    rsc2 = RDSE(parameter1)
    parameter2 = RDSE_Parameters()
    parameter2.size = 2000
    parameter2.sparsity = 0.02
    parameter2.resolution = 0.88
    rsc3 = RDSE(parameter2)
    parameter3 = RDSE_Parameters()
    parameter3.size = 2000
    parameter3.sparsity = 0.02
    parameter3.resolution = 0.88
    rsc1 = RDSE(parameter3)
    for _x1, _x2, _x3 in zip(col1, col2, col3):
        x_x1 = rsc1.encode(_x1)
        x_x2 = rsc2.encode(_x2)
        x_x3 = rsc3.encode(_x3)
        pooler_data.append(SDR(6000).concatenate([x_x1, x_x2, x_x3]))
    return pooler_data

def scaler_data():
    pooler_data = []
    data = pickle.load(open('data/x_hat.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    parameters1 = ScalarEncoderParameters()
    parameters1.minimum = np.min(col1)
    parameters1.maximum = np.max(col1)
    parameters1.size = 2000
    parameters1.sparsity = 0.02
    sc1 = ScalarEncoder(parameters1)
    parameters2 = ScalarEncoderParameters()
    parameters2.minimum = np.min(col2)
    parameters2.maximum = np.max(col2)
    parameters2.size = 2000
    parameters2.sparsity = 0.02
    sc2 = ScalarEncoder(parameters2)
    parameters3 = ScalarEncoderParameters()
    parameters3.minimum = np.min(col3)
    parameters3.maximum = np.max(col3)
    parameters3.size = 2000
    parameters3.sparsity = 0.02
    sc3 = ScalarEncoder(parameters3)
    for _x1, _x2, _x3 in zip(col1, col2, col3):
        x_x1 = sc1.encode(_x1)
        x_x2 = sc2.encode(_x2)
        x_x3 = sc3.encode(_x3)
        pooler_data.append(SDR(6000).concatenate([x_x1, x_x2, x_x3]))
    return pooler_data


def spatial_pooler_encoer(pooler_data):
    sp1 = SpatialPooler(
        inputDimensions=(6000,),
        columnDimensions=(6000,),
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=0.0435,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13999999999999999,
        boostStrength=3.0,
        wrapAround=True
    )
    sp2 = SpatialPooler(
        inputDimensions=(6000,),
        columnDimensions=(6000,),
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=0.0235,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13999999999999999,
        boostStrength=3.0,
        wrapAround=True
    )
    sp_info = Metrics(sp1.getColumnDimensions(), 999999999)
    base_or = np.zeros(6000).reshape((3, 2000)).astype(np.int8)
    sdr_array1 =[]
    sdr_array2 =[]
    for encoding in pooler_data:
        activeColumns1 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns1)
        sdr_array1.append(activeColumns1)
        # sp_info.addData(activeColumns)
    for encoding in sdr_array1:
        activeColumns2 = SDR(sp1.getColumnDimensions())
        sp2.compute(encoding, True, activeColumns2)
        sdr_array2.append(activeColumns2)
        # sp_info.addData(activeColumns)
    hold_out1 = sdr_array2[10001]
    hold_out2 = sdr_array2[10002]
    (x_train, _), (x_test, _) = mnist.load_data()
    anm = np.asarray(PIL.Image.open('images/anm.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm1 = np.asarray(PIL.Image.open('images/anm1.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    anm2 = np.asarray(PIL.Image.open('images/anm2.jpg').convert('L')).reshape((1, 28, 28, 1)) / (255 - 1)
    x_train = x_train.astype('float32') / (255 - 1)
    x_test = x_test.astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.concatenate((x_test, anm, anm1, anm2))
    counter =0
    for _s ,_img in zip( sdr_array2, x_test):
        # print(floor(_s.getSparsity()*6000))
        _x_ = hold_out2.getOverlap(_s)
        if _x_ > 113:
            _img_  = _img.reshape((28,28))
            _img_ =(_img_ * 254).astype(np.uint8)
            im = Image.fromarray(_img_).convert('RGB')
            im.save('test_results/'+str(counter)+'outfile.jpg')
            print('Sparsity - '+ str(_s.getSparsity()))
            print(_x_)
            print(str('counter - ') + str(counter))
            counter += 1



def main():
    # model = auto_encs()
    # new_model = pre_data(model)
    # save_model(new_model)
    # new_model = load_model()
    # predict_and_reduce(new_model)
    pooler_data = sacler_data_randonscaler()
    spatial_pooler_encoer(pooler_data)
    strs = ' '


if __name__ == '__main__':
    main()
    """    for pool1 in pooler_data2:
        for pool2 in pooler_data:
            _x_ =pool1.getOverlap(pool2)
            if _x_ > 118:
                print(_x_)
                counter +=1
                print(str('counter - ') + str(counter))
"""
