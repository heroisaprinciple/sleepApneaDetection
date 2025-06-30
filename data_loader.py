import numpy as np
import tensorflow as tf
import os

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

class DataLabel:
    def __init__(self, dir):
        self.dir = dir
        self.paths = []
        self.labels = []

    def label_data(self):
        patients = os.listdir(self.dir)
        for patient in patients:
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


if __name__ == "__main__":
    save_dir = "D:/apneaSpectrograms"

    labeler = DataLabel(save_dir)
    paths, labels = labeler.label_data()
    print(paths)
    print(labels) # [1 1 1 ... 0 0 0]

    file_paths = tf.constant(paths) # creates a paths tensor (tf.Tensor([b'[path]/seg_000.npy')
    file_labels = tf.constant(labels) # creates a labels tensor (tf.Tensor([1 1 1 ... 0 0 0])

    wrapper = NumpyWrapper()
    spectrogram, label = wrapper.tf_load_npy(file_paths[0], file_labels[0])
    print("Shape:", spectrogram.shape)
    print("Ten seconds:", spectrogram[:, :313])

    ds_creator = DatasetCreation(wrapper)
    print(ds_creator.create_dataset(file_paths, file_labels))

    ds = ds_creator.create_dataset(file_paths, file_labels)
    for spec, lab in ds.take(1):
        print("Batch spectrograms shape:", spec.shape)
        print("Batch labels:", lab.numpy()) # [1...1]

    # non-apnea
    # find index of first non-apnea segment
    non_apnea_idx = np.where(labels == 0)[0][0]
    path = file_paths[non_apnea_idx]
    label = file_labels[non_apnea_idx]
    spectrogram, label = wrapper.tf_load_npy(path, label)

    print(non_apnea_idx) # first non-apnea idx: 212
    print("Label:", label.numpy())  # 0
    print("Shape:", spectrogram.shape) # Shape: (64, 313)





