import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

class DataLabel:
    def __init__(self, dir, patients_list):
        self.dir = dir
        self.patients_list = patients_list
        self.paths = []
        self.labels = []

    def label_data(self):
        for patient in self.patients_list:
            patient_path = os.path.join(self.dir, patient) # [path]\00001444-100507
            if not os.path.isdir(patient_path): # if not a dir, pass
                continue

            for label, val in [("apnea", 1), ("non_apnea", 0)]:
                folder = os.path.join(patient_path, label)
                if not os.path.exists(folder): continue

                for filename in os.listdir(folder):
                    if filename.endswith(".npy"):
                        self.paths.append(os.path.join(folder, filename)) # [path]\apnea\seg_198.npy
                        self.labels.append(val)

        return np.array(self.paths), np.array(self.labels)

class NumpyWrapper:
    # tensorflow does not natively support .npy files
    # so a wrapper is used here to load them via numpy inside tensorflow
    def load_npy(self, file_path):
        spectrogram = np.load(file_path.numpy())
        return spectrogram.astype(np.float32)

    def tf_load_npy(self, file_path, label):
        # wrap the numpy loader using tf.py_function so TensorFlow can call it
        spectrogram = tf.py_function(func=self.load_npy, inp=[file_path], Tout=tf.float32)
        spectrogram.set_shape([64, 313])
        spectrogram = tf.expand_dims(spectrogram, axis=-1)  # shape: (64, 313, 1)
        return spectrogram, label

class DatasetCreation:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def create_dataset(self, files, labels):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        # load and preprocess multiple samples in parallel
        ds = ds.map(self.wrapper.tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
        # start loading the next batch while the model is still training on the current one => makes training faster
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

class DatasetSplitter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

        # .seed introduces a fixed split
        patients = sorted([p for p in os.listdir(self.save_dir)
                           if os.path.isdir(os.path.join(save_dir, p))])
        np.random.seed(0) # random number
        np.random.shuffle(patients)

        n = len(patients)
        train_patients = patients[:int(n * 0.8)] # 80% of patients
        val_patients = patients[int(n * 0.8):int(0.9 * n)] # 10% of patients
        test_patients = patients[int(n * 0.9):] # 10% of patients

        self.train_paths, self.train_labels = DataLabel(save_dir, train_patients).label_data()
        self.val_paths, self.val_labels = DataLabel(save_dir, val_patients).label_data()
        self.test_paths, self.test_labels = DataLabel(save_dir, test_patients).label_data()

        print(self.train_paths)
        print(self.train_labels)  # [1 1 1 ... 0 0 0]

# TODO: solve underfitting problem
class CNNBuilder:
    def build_cnn(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(64, 313, 1))) # only one channel: amplitude in db

        # conv layer 1
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # conv layer 2
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # cov layer 3
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # .flatten would lead to more parameters, so .globAvgPooling would be better
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

class GraphBuilder:
    @staticmethod
    def build_accuracy_graph(history):
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

class ConfusionMatrix:
    @staticmethod
    def build_confusion_matrix(test_ds, cnn_model):
        y_true = []
        y_pred = []
        for x_batch, y_batch in test_ds:
            # x_batch is a spectrogram batch -> shape: (64, 64, 313, 1)
            # y_batch is a batch of true labels -> shape: (64,)
            preds = cnn_model.predict(x_batch) # returns a numpy arr of apnea probabilities

            # convert probabilities to class labels (0 or 1); if prob < 0.5 => False (0)
            preds = (preds > 0.5).astype(int).flatten() # flatten because conf matrix expects 1D vectors

            y_true.extend(y_batch.numpy().astype(int).flatten()) # a numpy arr of ground truth values
            y_pred.extend(preds) # preds is already a numpy arr

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

if __name__ == "__main__":
    save_dir = "D:/apneaSpectrograms"

    splitter = DatasetSplitter(save_dir)
    wrapper = NumpyWrapper()
    ds_creator = DatasetCreation(wrapper)

    train_paths = tf.constant(splitter.train_paths) # creates a paths tensor (tf.Tensor([b'[path]/seg_000.npy')
    train_labels = tf.constant(splitter.train_labels) # creates a labels tensor (tf.Tensor([1 1 1 ... 0 0 0])
    val_paths = tf.constant(splitter.val_paths)
    val_labels = tf.constant(splitter.val_labels)
    test_paths = tf.constant(splitter.test_paths)
    test_labels = tf.constant(splitter.test_labels)

    train_ds = ds_creator.create_dataset(train_paths, train_labels)
    val_ds = ds_creator.create_dataset(val_paths, val_labels)
    test_ds = ds_creator.create_dataset(test_paths, test_labels)

    for spec, lab in train_ds.take(1):
        print("Train Batch spectrograms shape:", spec.shape) # (64, 64, 313) => a full batch of 64 spectrograms
        print("Train Batch labels:", lab.numpy()) # [0...0]

    for spec, lab in val_ds.take(1):
        print("Val Batch spectrograms shape:", spec.shape)  # (64, 64, 313) => a full batch of 64 spectrograms
        print("Val Batch labels:", lab.numpy()) # [1...1]

    for spec, lab in test_ds.take(1):
        print("Test Batch spectrograms shape:", spec.shape) # (64, 64, 313) => a full batch of 64 spectrograms
        print("Test Batch labels:", lab.numpy()) # [0...1]

    # load one sample from the first file
    spectrogram, label = wrapper.tf_load_npy(train_paths[0], train_labels[0])
    print("Shape:", spectrogram.shape)
    print("Ten seconds:", spectrogram[:, :313])

    # non-apnea
    # find index of first non-apnea segment
    non_apnea_idx = np.where(splitter.train_labels == 0)[0][0]
    path = tf.constant(splitter.train_paths[non_apnea_idx])
    label = tf.constant(splitter.train_labels[non_apnea_idx])

    spectrogram, label = wrapper.tf_load_npy(path, label)
    print(non_apnea_idx) # first non-apnea idx: 305
    print("Label:", label.numpy())  # 0
    print("Shape:", spectrogram.shape) # Shape: (64, 313)  => one spectrogram

    cnn_builder = CNNBuilder()
    cnn_model = cnn_builder.build_cnn()
    cnn_model.summary()

    # train model
    history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=10)
    loss, acc = cnn_model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")

    # build a graph
    graph = GraphBuilder.build_accuracy_graph(history)

    # conf matrix
    confusion_matrix = ConfusionMatrix.build_confusion_matrix(test_ds, cnn_model)







