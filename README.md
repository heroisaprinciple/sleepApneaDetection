# Sleep Apnea Detection with Deep Learning (TensorFlow + GPU)

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

## Data Engineering
- Added standardization (was min-max normalization before)

---

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sleepApneaDetection.git
cd sleepApneaDetection
````

### 2. CUDA and cuDNN Installation
For the guide, please refer [here](https://www.tensorflow.org/install/pip)

### 3. Run all packages in the project
````pip install -r requirements.txt````

---

## References
- Kaggle Dataset [used](https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios/data).
  The author: Bryan Darquea

This project is licensed under the MIT License.  
Please note that removing or altering the copyright notice is strictly prohibited and constitutes a violation of copyright law.