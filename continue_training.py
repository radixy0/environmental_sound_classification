import os
import datetime

import model_architecture

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from train import getData

weights_file="model/weights"

num_classes = 10
learning_rate = 1e-4
decay = 1e-6
momentum = 0.9
epochs = 250
batch_size = 16


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

model = model_architecture.model1(num_classes, input_shape)
print("loaded model: ", model.name)
load_status = model.load_weights(weights_file)
load_status.expect_partial()

sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks=[
    keras.callbacks.EarlyStopping(patience=2, verbose=1),
    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    keras.callbacks.ModelCheckpoint(
            weights_file, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch',
            options=None
        )
]

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)
