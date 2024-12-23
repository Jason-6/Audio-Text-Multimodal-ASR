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

## Contact
For questions or support, please contact:
- [Your Name](mailto:your.email@example.com)
- [Repository Issues](https://github.com/yourusername/Audio-Text-Multimodal-ASR/issues)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Abstract**:

In air traffic control communications (ATCC), misunderstandings between pilots and controllers could result in fatal aviation accidents. Fortunately, advanced automatic speech recognition technology has emerged as a promising means of preventing miscommunications and enhancing aviation safety. However, most existing speech recognition methods merely incorporate external language models on the decoder side, leading to insufficient semantic alignment between speech and text modalities during the encoding phase. Furthermore, it is challenging to model acoustic context dependencies over long distances due to the longer speech sequences than text, especially for the extended ATCC data. To address these issues, we propose a speech-text multimodal dual-tower architecture for speech recognition. It employs cross-modal interactions to achieve close semantic alignment during the encoding stage and strengthen its capabilities in modeling auditory long-distance context dependencies. In addition, a two-stage training strategy is elaborately devised to derive semantics-aware acoustic representations effectively. The first stage focuses on pre-training the speech-text multimodal encoding module to enhance inter-modal semantic alignment and aural long-distance context dependencies. The second stage fine-tunes the entire network to bridge the input modality variation gap between the training and inference phases and boost generalization performance. Extensive experiments demonstrate the effectiveness of the proposed speech-text multimodal speech recognition method on the ATCC and AISHELL-1 datasets. It reduces the character error rate to 6.54% and 8.73%, respectively, and exhibits substantial performance gains of 28.76% and 23.82% compared with the best baseline model. The case studies indicate that the obtained semantics-aware acoustic representations aid in accurately recognizing terms with similar pronunciations but distinctive semantics. The research provides a novel modeling paradigm for semantics-aware speech recognition in air traffic control communications, which could contribute to the advancement of intelligent and efficient aviation safety management.
