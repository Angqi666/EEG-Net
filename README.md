# EEGNet Replication

This repository contains a replication of the EEGNet code, an efficient convolutional neural network architecture for EEG-based brain–computer interface applications. The project is intended to make the original work more accessible and to serve as a basis for further research and experimentation in EEG signal analysis.

## Overview

**EEGNet** is a compact CNN architecture originally developed by Lawhern et al. for classifying EEG signals in various BCI tasks. This repository includes:
- **Data Preprocessing:** Scripts for loading and processing EEG data.
- **Model Architecture:** Implementation of the EEGNet neural network.
- **Training and Evaluation:** Pipelines to train the network and evaluate its performance.

## Citation

If you use this code or its ideas in your research, please cite the original work:

> Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering, 15(5), 056013.

## Installation

### Requirements

- Python 3.10.14
- Pytorch 2.5.1

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/EEGNet-replication.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd EEGNet-replication
   ```

## Usage

### Training

To start training the model, run:
```bash
python train.py 
```

## Project Structure

- **`/data_loader.py`**: you need to have you own data and use the path of the data in the file (e.g. shape:(1,eeg channels, sample pts)).
- **`/models.py`**: Definition of the EEGNet architecture. You can change the F1 and D as you want 
- **`/train.py`**: Training and evaluation pipelines.
- **`README.md`**: This file.


## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.


## Acknowledgements

Many thanks to the original authors for their groundbreaking work on EEGNet, which has inspired this replication. Their contributions are acknowledged through proper citation and reference in this repository.

