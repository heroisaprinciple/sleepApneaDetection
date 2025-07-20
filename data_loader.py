import numpy as np
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import random

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000
SEED = 42

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
    def __init__(self, training = True):
        self.training = training

    # tensorflow does not natively support .npy files
    # so a wrapper is used here to load them via numpy inside tensorflow
    def load_npy(self, file_path):
        spectrogram = np.load(file_path.numpy())
        # add standardization
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = (spectrogram - mean) / (std + 1e-8)

        # augment only while training
        if self.training and np.random.rand() < 0.5: # augment for 50% of samples
            spectrogram = self.spec_augment(spectrogram)
        return spectrogram.astype(np.float32)

    def tf_load_npy(self, file_path, label):
        # wrap the numpy loader using tf.py_function so TensorFlow can call it
        spectrogram = tf.py_function(func=self.load_npy, inp=[file_path], Tout=tf.float32)
        spectrogram.set_shape([64, 313])
        spectrogram = tf.expand_dims(spectrogram, axis=-1)  # shape: (64, 313, 1)
        return spectrogram, label

    # data augmentation - mask or zero out a random block of time steps
    # choose a random point and zero out n time steps columns -> x axis
    def spec_augment(self, spectrogram, time_masking_max=10, freq_masking_max=3):
        # time masking
        t = spectrogram.shape[1] # 313 time steps
        t0 = np.random.randint(0, t - time_masking_max)
        spectrogram[:, t0:t0 + time_masking_max] = 0.0 # set n columns to 0

        # frequency bins masking
        # same concept - choose a random point and zero out n frequency bins -> y axis
        f = spectrogram.shape[0] # 64
        f0 = np.random.randint(0, f - freq_masking_max)
        spectrogram[f0:f0 + freq_masking_max, :] = 0.0 # set n rows to 0
        return spectrogram

class DatasetCreation:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def create_dataset(self, files, labels):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        # load and preprocess multiple samples in parallel
        ds = ds.map(self.wrapper.tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
        # start loading the next batch while the model is still training on the current one => makes training faster
        # TODO: see if seed is good here
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

    @staticmethod
    def build_loss_graph(history):
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

class ConfusionMatrix:
    @staticmethod
    def build_confusion_matrix(test_ds, cnn_model):
        y_true = []
        y_pred = []
        y_pred_probs = [] # for roc auc curve
        for x_batch, y_batch in test_ds:
            # x_batch is a spectrogram batch -> shape: (64, 64, 313, 1)
            # y_batch is a batch of true labels -> shape: (64,)
            preds = cnn_model.predict(x_batch) # returns a numpy arr of apnea probabilities

            # convert probabilities to class labels (0 or 1); if prob < 0.5 => False (0)
            preds = (preds > 0.5).astype(int).flatten() # flatten because conf matrix expects 1D vectors

            y_true.extend(y_batch.numpy().astype(int).flatten()) # a numpy arr of ground truth values
            y_pred.extend(preds) # preds is already a numpy arr
            y_pred_probs.extend(preds)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        return np.array(y_true), np.array(y_pred), np.array(y_pred_probs)

    @staticmethod
    def add_metrics(y_true, y_pred):
        # there is no accuracy here as it is the same as the test accuracy printed in __main__
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        f1 = f1_score(y_true, y_pred)

        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("Precision:", precision)
        print("F1:", f1)

        print("Classification Report:")
        print(classification_report(y_true, y_pred))

    @staticmethod
    def add_roc_auc_curve(y_true, y_pred):
        # false positive rate, true positive rate and threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print("ROC AUC:", roc_auc)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='red', lw=2, label=f'AUC = {roc_auc}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate (Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

class ClassWeights:
    @staticmethod
    def find_class_weights(splitter):
        train_labels = splitter.train_labels
        weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: weights[i] for i in range(len(weights))}
        print("Class weights:", class_weights_dict)
        return class_weights_dict

class EarlyStop:
    @staticmethod
    def add_early_stop(metrics, patience):
        return EarlyStopping(monitor=metrics, patience=patience, restore_best_weights=True)

class ModelCheckpointer:
    @staticmethod
    def get_model_checkpoint(path, metrics):
        return ModelCheckpoint(path, monitor=metrics, save_best_only=True, save_weights_only=False)

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    save_dir = "D:/apneaSpectrograms"
    best_model_path = "models/best_model.keras"

    splitter = DatasetSplitter(save_dir)
    train_wrapper = NumpyWrapper(training=True)
    eval_wrapper = NumpyWrapper(training=False)

    ds_creator_train = DatasetCreation(train_wrapper)
    ds_creator_eval = DatasetCreation(eval_wrapper)

    train_paths = tf.constant(splitter.train_paths) # creates a paths tensor (tf.Tensor([b'[path]/seg_000.npy')
    train_labels = tf.constant(splitter.train_labels) # creates a labels tensor (tf.Tensor([1 1 1 ... 0 0 0])
    val_paths = tf.constant(splitter.val_paths)
    val_labels = tf.constant(splitter.val_labels)
    test_paths = tf.constant(splitter.test_paths)
    test_labels = tf.constant(splitter.test_labels)

    train_ds = ds_creator_train.create_dataset(train_paths, train_labels)
    val_ds = ds_creator_eval.create_dataset(val_paths, val_labels)
    test_ds = ds_creator_eval.create_dataset(test_paths, test_labels)

    print("Train label counts:", np.bincount(splitter.train_labels))
    print("Val label counts:", np.bincount(splitter.val_labels))
    print("Test label counts:", np.bincount(splitter.test_labels))

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
    spectrogram, label = train_wrapper.tf_load_npy(train_paths[0], train_labels[0])
    print("Shape after training and augmentation:", spectrogram.shape)
    print("Ten seconds:", spectrogram[:, :313])

    # non-apnea
    # find index of first non-apnea segment
    non_apnea_idx = np.where(splitter.train_labels == 0)[0][0]
    path = tf.constant(splitter.train_paths[non_apnea_idx])
    label = tf.constant(splitter.train_labels[non_apnea_idx])

    # training
    spectrogram, label = train_wrapper.tf_load_npy(path, label)
    print(non_apnea_idx)  # first non-apnea idx: 305
    print("Label:", label.numpy())  # 0
    print("Shape after training and augmentation:", spectrogram.shape)  # Shape: (64, 313)  => one spectrogram

    cnn_builder = CNNBuilder()
    cnn_model = cnn_builder.build_cnn()
    cnn_model.summary()

    class_weights = ClassWeights.find_class_weights(splitter)

    # do early stop after 15 epochs if validation loss is not decreasing
    early_stop = EarlyStop.add_early_stop(metrics='val_loss', patience=15)

    # add model checkpoint
    model_checkpoint = ModelCheckpointer.get_model_checkpoint(path=best_model_path, metrics='val_loss')

    # train model
    history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=20, class_weight=class_weights,
                            callbacks=[early_stop, model_checkpoint])

    loss, acc = cnn_model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")

    print("Stopped at epoch:", len(history.history['loss']))

    # build a graph
    GraphBuilder.build_accuracy_graph(history)
    GraphBuilder.build_loss_graph(history)

    # conf matrix and metrics
    y_true, y_pred, y_pred_probs = ConfusionMatrix.build_confusion_matrix(test_ds, cnn_model)
    ConfusionMatrix.add_metrics(y_true=y_true, y_pred=y_pred)
    ConfusionMatrix.add_roc_auc_curve(y_true=y_true, y_pred=y_pred_probs)
