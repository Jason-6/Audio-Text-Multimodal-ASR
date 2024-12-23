# Audio-Text-Multimodal-ASR

This repository contains the code and resources for the paper:

**Audio-Text Multimodal Speech Recognition via Dual-Tower Architecture for Mandarin Air Traffic Control Communications**

## Introduction
In air traffic control communications (ATCC), miscommunications between pilots and controllers can lead to serious aviation accidents. This work introduces a novel speech-text multimodal dual-tower architecture designed to address these issues by:

- **Semantic Alignment**: Enhancing semantic alignment between speech and text during the encoding phase.
- **Modeling Long-Distance Context**: Strengthening the ability to model long-distance acoustic dependencies in extended ATCC data.

The architecture employs:
1. Cross-modal interactions for close semantic alignment during encoding.
2. A two-stage training strategy to:
   - Pre-train the multimodal encoding module.
   - Fine-tune the entire network to bridge the modality gap and boost performance.

The method demonstrates significant improvements over existing speech recognition techniques.

## Key Features
- **Dual-Tower Architecture**: Incorporates audio and text modalities for joint semantic representation.
- **Two-Stage Training**: 
  - Stage 1: Enhances inter-modal semantic alignment and auditory long-distance context modeling.
  - Stage 2: Fine-tunes the model for improved generalization.
- **Improved Recognition Accuracy**: Achieves notable reductions in character error rate (CER) on benchmark datasets.

## Results
- **Datasets**:
  - ATCC
  - AISHELL-1
- **Performance**:
  - CER on ATCC: 6.54%
  - CER on AISHELL-1: 8.73%
  - Performance gains compared to baselines:
    - ATCC: +28.76%
    - AISHELL-1: +23.82%

## Usage
### Prerequisites
- Python >= 3.8
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Audio-Text-Multimodal-ASR.git
   cd Audio-Text-Multimodal-ASR
   ```
2. **Prepare Data**:
   - Place ATCC and AISHELL-1 datasets in the `data/` directory.
   - Preprocess the data using:
     ```bash
     python preprocess.py
     ```
3. **Train the Model**:
   - Stage 1 (Pre-training):
     ```bash
     python train.py --stage 1
     ```
   - Stage 2 (Fine-tuning):
     ```bash
     python train.py --stage 2
     ```
4. **Evaluate**:
   ```bash
   python evaluate.py --checkpoint <model_checkpoint>
   ```

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@Article{GE20243215,
AUTHOR = {Shuting Ge, Jin Ren, Yihua Shi, Yujun Zhang, Shunzhi Yang, Jinfeng Yang},
TITLE = {Audio-Text Multimodal Speech Recognition via Dual-Tower Architecture for Mandarin Air Traffic Control Communications},
JOURNAL = {Computers, Materials \& Continua},
VOLUME = {78},
YEAR = {2024},
NUMBER = {3},
PAGES = {3215--3245},
URL = {http://www.techscience.com/cmc/v78n3/55898},
ISSN = {1546-2226},
DOI = {10.32604/cmc.2023.046746}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
