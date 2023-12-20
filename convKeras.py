import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense

import os
import matplotlib.pyplot as plt

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
    ]
)


def filter_images():
    num_skipped = 0
    for folder_name in ("train\happy", "train\\angry"):
        folder_path = os.path.join(".\images", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)


def generate_dataset():
    image_size = (48, 48)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ".\images",
        validation_split=0.2,
        subset="training",
        seed=133,  #ZMIENIC TEGO SEEDA
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ".\images",
        validation_split=0.2,
        subset="validation",
        seed=133,
        image_size=image_size,
        batch_size=batch_size,
    )

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

    train_ds = train_ds.prefetch(buffer_size=32)    #Przyspiesza ładowanie i operacje na danych
    val_ds = val_ds.prefetch(buffer_size=32)


    model = make_model(input_shape=image_size + (3,), num_classes=2)
    #keras.utils.plot_model(model, show_shapes=True)

    epochs = 1      #liczba iteracji

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),  #zapisywanie Modelu danego przejścia
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),  #learning rate, korzystamy z algorytmu Adam ze wzgledu na niski koszt treningu dla wiekszych modeli i wiekszej ilosci iteracji, od 50 juz widac mocna roznice
        loss="binary_crossentropy", #Algorytm wyznaczający stratę pomiędzy prawdziwymi wartosciami a predykcją
        metrics=["accuracy"],   #Jak czesto predykcje są zgodne z prawdą
        #Zalenosci pomiedzy accuracy a loss są, ale nie zawsze musza byc dokladne
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    img = keras.preprocessing.image.load_img(
        ".\images\\train\\happy\Training_177442.jpg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)     #3 wymiarowa tablica
    img_array = tf.expand_dims(img_array, 0)  #Dodanie wymiaru początkowego do pojedynczego elementu !!!OGARNAC JESZCZE

    predictions = model.predict(img_array)  #obliczana jest predykcja wyników
    score = predictions[0]
    print(
        "This image is %.2f percent happy and %.2f percent angry."
        % (100 * (1 - score), 100 * score)
    )




def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = keras.layers.Rescaling(1.0 / 255)(x)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(x) # 2D convolution layer; 32 - liczba filtrów wyjściowych; 3 - maska 3x3; strides=2; padding="same" dopełnienie zerami
    x = keras.layers.BatchNormalization()(x) # Normalizacja wsadowa stosuje transformację, która utrzymuje średnią wartość wyjściową bliską 0, a odchylenie standardowe wyjścia bliskie 1
    x = keras.layers.Activation("relu")(x) # funkcja aktywacji relu

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x) # Dzielimy całości na kwadraty 2x2 i wybieramy największą wartośc (ogólnie redukcja danych), prop. wsteczna CHRONI PRZED PRZEUCZENIEM

        # Project residual
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # SeparableConv2D: Rozdzielne sploty polegają na wykonaniu najpierw głębokiego splotu
    # przestrzennego(który działa na każdy kanał wejściowy osobno), a następnie
    # splotu punktowego, który miesza powstałe kanały wyjściowe. Liczba filtrow/ wysokosc i szerokosc tablicy konwulsji 2d same= wyniki z rownomiernie rozmieszczonymi zerami są takie same
    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x) # operacja łączenia danych przestrzennych

    # wybór sposobu aktywacji ostatniej warstwy (wartości na ostatniej warstwie sumują się do 1)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def run_test():

    image_size = (48, 48)

    model = keras.models.load_model("save_at_1.h5")

    img = keras.preprocessing.image.load_img(
        ".\images\\train\\happy\Training_177442.jpg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent happy and %.2f percent angry."
        % (100 * (1 - score), 100 * score)
    )

#filter_images()
generate_dataset()
run_test()