import numpy as np
import model_architecture
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

weights_file = "model/weights"
x_path = "data/x_train.npy"
y_path = "data/y_train.npy"

x_val_path = "data/x_val.npy"
y_val_path = "data/y_val.npy"

num_classes = 10
learning_rate = 1e-3
decay = 1e-6
momentum = 0.9
epochs = 250
batch_size = 16


def getData():
    # get training files
    print("found saved training files")
    x_train = np.load(x_path)
    y_train = np.load(y_path)

    # get validation files
    print("found saved validation files")
    x_val = np.load(x_val_path)
    y_val = np.load(y_val_path)

    return x_train, y_train, x_val, y_val


def main():
    if not (os.path.isfile(x_path)) \
            or not (os.path.isfile(y_path)) \
            or not (os.path.isfile(x_val_path)) \
            or not (os.path.isfile(y_val_path)):
        print("files not found! aborting..")
        return

    x_train, y_train, x_val, y_val = getData()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    y_val = keras.utils.to_categorical(y_val, num_classes)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    print("x shape", x_train.shape)
    print("y shape: ", y_train.shape)
    print("x val shape: ", x_val.shape)
    print("y val shape: ", y_val.shape)
    print("test: ", x_train[333][20][21][0])  # test random value to see if its in [0,1]

    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(x_val))

    model = model_architecture.model1(10, input_shape)
    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = "logs/fit/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks=[
        keras.callbacks.EarlyStopping(patience=2, verbose=1),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            "model/weights", monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch',
            options=None
        )
    ]

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=2, callbacks=callbacks)

    print("\n\nAll done! Weights saved to "+weights_file)

if __name__ == "__main__":
    main()
