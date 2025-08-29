or# Sleep Apnea Detection with Deep Learning (TensorFlow + GPU)

This project uses a Convolutional Neural Network (CNN) model to detect sleep apnea events from audio spectrograms.  
All training and inference are accelerated using an NVIDIA GPU (via TensorFlow) on Windows.

---

## Environment

- **OS:** Windows 11 
- **NVIDIA GPU:** (RTX 3050)
- **Python:** 3.10
- **Conda** 
- **TensorFlow:** 2.10.1 
- **CUDA Toolkit:** 11.7
- **cuDNN:** 8.4.0
- **Other Python packages:**  
  - `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `librosa`

---

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sleepApneaDetection.git
cd sleepApneaDetection
````

### 2. CUDA and cuDNN Installation
For the guide, please refer [here.](https://www.tensorflow.org/install/pip)

### 3. Download the dataset, used in References. 
This is a preprocessed version of PSG-Audio with only Mic channel extracted.
The data is sliced to 10-second segments and resampled to 16kHz. 
After that, update paths to directories.

### 4. Run all packages in the project
````pip install -r requirements.txt````

### 5. Create a spectrogram dataset
Run ````convert_to_spectrograms_and_save.py```` to create a spectrogram dataset of a preprocessed version of PSG-Audio.
That must happen for the CNN to learn. 

### 6. Start data augmentation and model training and evaluation
Run ````model_train.py```` for the CNN to learn and evaluate the unseen data of segments.

### 7. Record 10-second segments with Raspberry Pi
Go to the [separate repository](https://github.com/heroisaprinciple/apneaDetectionRPi).
The README.md contains all the necessary instructions. Please, don't forget to create
your own venv.

### 8. Login to AWS as admin to view the uploaded recorded segment (apnearecordings is private for now)
In order to access it, your root account should transfer you the admin credentials.
Those must be saved as env variables on OS and never exposed in code.
All recordings will be saved under <i>apnearecordings/PATIENT_ID/YY/MM/DD/H/M/S</i> prefix.
The example:

![S3 Example](https://i.imgur.com/t0vFI7I.png)

**If you want to play with cloud, you might want to create a separate bucket yourself.**
Then, feel free to change the ````fetch_data.py```` file.

### 9. Fetch data
Run ````fetch_data.py```` to retrieve segments. They will be stored in fetched_recordings/ dir. 
It is not possible unless using admin credentials. 
If you created your own bucket, no worries.

### 10. Analyze original and recorded segments
Run ```extract_apnea_spectrograms.ipynb``` to create a spectrogram from a recorded segment.
The spectrogram for the original audio is also fetched there from a local directory.
Both original and recorded spectrograms are saved in fetched_recordings. 
Moreover, there are waveforms for you to see.
Also, the dominant frequency values will be seen for both audio segments when .npy files
are converted to .wav segments.

### 11. Prediction!
Run ````apnea_prediction.py```` to predict event for a segment. If prediction is <= 0.5,
the event is normal. Otherwise -> apneic. 
In future, the threshold-based rule is planned to be removed as AHI (Apnea-Hypopnea Index)
is much reliable for analysis.

---

## References
- Kaggle Dataset [used](https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios/data).
  The author: Bryan Darquea

This project is licensed under the MIT License.  

Please note that removing or altering the copyright notice is strictly prohibited and constitutes a violation of copyright law.
