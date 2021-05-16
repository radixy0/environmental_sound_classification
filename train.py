import numpy as np
import model_architecture
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

model_dir = "model/"
x_path = "data/x_train.npy"
y_path = "data/y_train.npy"

x_val_path = "data/x_val.npy"
y_val_path = "data/y_val.npy"

num_classes = 10
learning_rate = 0.0001
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
    if not (os.path.isfile(x_path)) or not (os.path.isfile(y_path)) or not (os.path.isfile(x_val_path)) or not (os.path.isfile(y_val_path)):
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

    model = model_architecture.VGG16_Untrained(10, input_shape)
    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, verbose=1)
    ]

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=1, callbacks=callbacks)

    model_filename = "model.h5"
    model.save(model_dir+model_filename, include_optimizer=True)

    print("\n\nAll done! Model and Diagrams saved to /model" + model_filename)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model/accuracy.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model/loss.jpg")


    #os.system("shutdown -s -t 30")


if __name__ == "__main__":
    main()
