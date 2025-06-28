import numpy as np
import tensorflow as tf
import os

class DataLabel:
    @staticmethod
    def label_data(dir):
        paths = []
        labels = []

        patients = os.listdir(dir)
        for patient in patients:
            patient_path = os.path.join(dir, patient) # D:/apneaSpectrograms\00001444-100507
            if not os.path.isdir(patient_path): # if not a dir, pass
                continue

            for label, val in [("apnea", 1), ("non_apnea", 0)]:
                folder = os.path.join(patient_path, label)
                if not os.path.exists(folder): continue

                for filename in os.listdir(folder):
                    if filename.endswith(".npy"):
                        paths.append(os.path.join(folder, filename)) # D:/apneaSpectrograms\00001488-100507\apnea\seg_198.npy
                        labels.append(val)

        return np.array(paths), np.array(labels)

if __name__ == "__main__":
    save_dir = "D:/apneaSpectrograms"

    paths, labels = DataLabel.label_data(save_dir)
    print(paths)
    print(labels) # [1 1 1 ... 0 0 0]


