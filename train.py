import numpy as np
import model_architecture
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import settings

def getData():
    # get training files
    print("found saved training files")
    x_train = np.load(settings.x_path)
    y_train = np.load(settings.y_path)

    # get validation files
    print("found saved validation files")
    x_val = np.load(settings.x_val_path)
    y_val = np.load(settings.y_val_path)

    return x_train, y_train, x_val, y_val


def main():
    if not (os.path.isfile(settings.x_path)) \
            or not (os.path.isfile(settings.y_path)) \
            or not (os.path.isfile(settings.x_val_path)) \
            or not (os.path.isfile(settings.y_val_path)):
        print("files not found! aborting..")
        return

    x_train, y_train, x_val, y_val = getData()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = keras.utils.to_categorical(y_train, settings.num_classes)

    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    y_val = keras.utils.to_categorical(y_val, settings.num_classes)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    print("x shape", x_train.shape)
    print("y shape: ", y_train.shape)
    print("x val shape: ", x_val.shape)
    print("y val shape: ", y_val.shape)
    print("test: ", x_train[333][20][21][0])  # test random value to see if its in [0,1]

    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(x_val))

    model = model_architecture.ResNet34(10, input_shape)
    sgd = SGD(lr=settings.learning_rate, decay=settings.decay, momentum=settings.momentum)
    adam = keras.optimizers.Adam(learning_rate=settings.learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = "logs/fit/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks=[
        keras.callbacks.EarlyStopping(patience=settings.patience, verbose=1),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            "model/model.h5", monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=settings.lr_patience,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-6
        )
    ]

    history = model.fit(x_train, y_train, epochs=settings.epochs, batch_size=settings.batch_size, validation_data=(x_val, y_val),
                        verbose=2, callbacks=callbacks)

    print("\n\nAll done! Weights saved to "+settings.model_file)

if __name__ == "__main__":
    main()
