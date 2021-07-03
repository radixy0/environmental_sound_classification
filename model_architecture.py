def model1(num_classes, input_shape):
    print("model 1")
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    #model architecture:
    model=models.Sequential(name="model1")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes,  activation=tf.nn.softmax))

    return model

def model2(num_classes, input_shape):
    print("model 2")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

    model = Sequential()

    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def model3(num_classes, input_shape):
    print("model 3")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # model architecture:
    model = models.Sequential(name="model1")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes, activation=tf.nn.softmax))

    return model

def model4(num_classes, input_shape):
    print("model 4")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # model architecture:
    model = models.Sequential(name="model1")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes, activation=tf.nn.softmax))

    return model

def VGG16_Untrained(num_classes, input_shape):
    print("vgg_16")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


    model = Sequential(name="VGG16")
    model.add(ZeroPadding2D((1,1), input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def VGG16_Pretrained(num_classes, input_shape):
    print("pretrained vgg16")
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.models import Model

    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    flatten = Flatten()
    new_layer2 = Dense(num_classes, activation='softmax', name='my_dense_2')

    inp2 = model.input
    out2 = new_layer2(flatten(model.output))

    model2 = Model(inp2, out2)

    return model2


def VGG19_Untrained(num_classes, input_shape):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Conv2D, Input
    from tensorflow.keras.layers import MaxPooling2D
    print("vgg19")
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='vgg19')

    return model


def ResNet18(num_classes, input_shape):
    from classification_models.tfkeras import Classifiers
    from tensorflow.keras import layers, Model

    ResNet18, _ = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=input_shape, include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[base_model.input], outputs=[output])


def ResNet34(num_classes, input_shape):
    from classification_models.tfkeras import Classifiers
    from tensorflow.keras import layers, Model

    ResNet34, _ = Classifiers.get('resnet34')
    base_model = ResNet34(input_shape=input_shape, include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[base_model.input], outputs=[output])


def ResNet50(num_classes, input_shape):
    from classification_models.tfkeras import Classifiers
    from tensorflow.keras import layers, Model

    resnet50, _ = Classifiers.get('resnet50')
    base_model = resnet50(input_shape=input_shape, include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[base_model.input], outputs=[output])

def ResNet50V2(num_classes, input_shape):
    from tensorflow.keras import applications, layers, Model
    model = applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="max",
        classes=num_classes,
        classifier_activation="softmax"
    )
    out = layers.Dense(num_classes, activation="softmax")

    return Model(inputs=model.input, outputs=out(model.output))

def ResNet101V2(num_classes, input_shape):
    from tensorflow.keras import applications, layers, Model
    model = applications.ResNet101V2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="max",
        classes=num_classes,
        classifier_activation="softmax"
    )
    out = layers.Dense(num_classes, activation="softmax")
    return Model(inputs=model.input, outputs=out(model.output))




def InceptionV3(num_classes, input_shape):
    from tensorflow.keras import applications, layers, Model
    model = applications.InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling="max",
        classes=num_classes,
        classifier_activation="softmax"
    )
    out=layers.Dense(num_classes, activation="softmax")
    return Model(inputs=model.input, outputs=out(model.output))

def model_paper(num_classes, input_shape):
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
    from tensorflow.keras import Model
    from tensorflow.keras.regularizers import l2

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # l1
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(24, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(spec_x)
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_last")(spec_x)

    # l2
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(spec_x)
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_last")(spec_x)

    # l3
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(spec_x)
    spec_x = BatchNormalization(axis=3)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Flatten()(spec_x)
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(spec_x)

    spec_x = Dropout(0.5)(spec_x)
    out = Dense(num_classes,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model

