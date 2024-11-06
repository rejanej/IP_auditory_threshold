import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

directory = './scalogram'

img_height = 375
img_width = 82
validation_split = 0.3

train = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=100)

valid = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=100)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
val_ds = valid.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

train_data = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return np.concatenate(images), np.concatenate(labels)

X_train_data, Y_train_data = dataset_to_numpy(train_data)
X_valid_data, Y_valid_data = dataset_to_numpy(val_data)

# Verify shapes of arrays
print(f'X_train shape: {X_train_data.shape}')
print(f'Y_train shape: {Y_train_data.shape}')
print(f'X_valid shape: {X_valid_data.shape}')
print(f'Y_valid shape: {Y_valid_data.shape}')

with tf.device('/CPU:0'):
    tf.config.list_physical_devices()

    num_classes = 2
    labels = ['ABR_absent', 'ABR_present']
    Y_train = tf.keras.utils.to_categorical(Y_train_data, num_classes)
    X_train = X_train_data.reshape(X_train_data.shape[0], img_height, img_width, 3).astype('float32')
    Y_test = tf.keras.utils.to_categorical(Y_valid_data, num_classes=num_classes)
    X_test = X_valid_data.reshape(X_valid_data.shape[0], img_height, img_width, 3).astype('float32')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_data=(X_test, Y_test))

#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%

with tf.device('/CPU:0'):
    tf.config.list_physical_devices()
    Y_predict = model.predict(X_test)

#%%

Y_predict = model.predict(X_test)

i = 131
predicted_num = np.argmax(Y_predict[i])
print("Network output", Y_predict[i])
print("Max network output", np.max(Y_predict[i]))
print("Number to be predicted",Y_test[i])
print("Predicted number", predicted_num)
plt.title("{} {} {} {} {} {}".format('To be predicted :', str(Y_test[i]),
                                  'Predicted :', str(predicted_num),'|',
                                  str(np.max(Y_predict[i]))),
          color="green" if Y_test[i] == predicted_num else "red")

plt.imshow(X_test[i], cmap='Greys')
plt.show()