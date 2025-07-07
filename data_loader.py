import numpy as np
import tensorflow as tf
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
        spectrogram.set_shape([64, None])
        return spectrogram, label

class DatasetCreation:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def create_dataset(self, files, labels):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        ds = ds.map(self.wrapper.tf_load_npy)
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
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






