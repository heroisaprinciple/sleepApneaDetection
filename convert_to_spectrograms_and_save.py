import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

base_dir = "D:/apneaCleaned/PSG-AUDIO/APNEA_EDF"
save_dir = "D:/apneaSpectrograms"

sampling_rate = 16000

print(len(os.listdir(base_dir))) # 192
patients = os.listdir(base_dir)
for patient_id in patients:
    patient_path = os.path.join(base_dir, patient_id)
    print(patient_path)
    if not os.path.isdir(patient_path):
        continue

    apnea_path = os.path.join(patient_path, f"{patient_id}_ap.npy")
    non_apnea_path = os.path.join(patient_path, f"{patient_id}_nap.npy")

    for label, npy_file in [("apnea", apnea_path), ("non_apnea", non_apnea_path)]:
        save_patient_dir = os.path.join(save_dir, patient_id, label)
        os.makedirs(save_patient_dir, exist_ok=True)
        if not os.path.exists(npy_file):
            print(f" Missing file: {npy_file}")
            continue

        segments = np.load(npy_file)

        # each .npy file will contain one 2D NumPy array, which is the Mel spectrogram in decibel scale for one 10-second audio segment.
        for i, segment in enumerate(segments):
            spectrogram = librosa.feature.melspectrogram(y=segment, sr=sampling_rate, n_mels=64) # 512 hops by default
            print(spectrogram.shape) # 64 mel freq bins and 313 small windows
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max) # needs decibels
            save_path = os.path.join(save_patient_dir, f"seg_{i:03d}.npy")
            np.save(save_path, spectrogram_db)
            print(f"saved {save_path}")
