import numpy as np
import model_architecture
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import settings

def getData():
    # get training files
    print("found saved training files")
    x_train = np.load(settings.x_path)
    y_train = np.load(settings.y_path)

    return x_train, y_train


def main():
    if not (os.path.isfile(settings.x_path)) \
            or not (os.path.isfile(settings.y_path)):
        print("files not found! aborting..")
        return

    data, labels = getData()
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    labels = keras.utils.to_categorical(labels, settings.num_classes)

    input_shape = (data.shape[1], data.shape[2], 1)
    print("x shape", data.shape)
    print("y shape: ", labels.shape)

    assert not np.any(np.isnan(data))

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.25)

    model = model_architecture.ResNet50(10, input_shape)
    sgd = SGD(lr=settings.learning_rate) #, decay=settings.decay, momentum=settings.momentum)
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
        #keras.callbacks.ReduceLROnPlateau(
        #    monitor="val_loss",
        #    factor=0.5,
        #    patience=settings.lr_patience,
        #    verbose=1,
        #    mode="auto",
        #    min_delta=0.0001,
        #    cooldown=0,
        #    min_lr=1e-6
        #)
    ]

    history = model.fit(x_train, y_train, epochs=settings.epochs, batch_size=settings.batch_size, validation_data=(x_val, y_val),
                        verbose=2, callbacks=callbacks)

    print("\n\nAll done! Model saved to "+settings.model_file)

if __name__ == "__main__":
    main()
