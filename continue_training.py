import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from train import getData
import settings


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

model = keras.models.load_model(settings.model_file)
print("loaded model: ", model.name)

sgd = SGD(lr=settings.learning_rate, decay=settings.decay, momentum=settings.momentum, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks=[
    keras.callbacks.EarlyStopping(patience=2, verbose=1),
    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    keras.callbacks.ModelCheckpoint(
            settings.model_file, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )
]

history = model.fit(x_train, y_train, epochs=settings.epochs, batch_size=settings.batch_size, validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)
