# Reproduction: A Hybrid Vision Transformer Approach for Mathematical Expression Recognition

## Reference
**Title:** A Hybrid Vision Transformer Approach for Mathematical Expression Recognition  
**Authors:** Anh Duy Le, Van Linh Pham, Vinh Loi Ly, Nam Quan Nguyen, Huu Thang Nguyen, Tuan Anh Tran  
**Paper Link:** [arXiv:2603.07929](https://arxiv.org/abs/2603.07929)  

## Method Description
This repository contains an independent reproduction of the **Hybrid Vision Transformer (HVT)** architecture proposed by Le et al. for image-to-LaTeX mathematical expression recognition. Translating rendered math equations back into LaTeX is a highly complex sequence-to-sequence problem due to the two-dimensional nature of mathematical syntax and severe variations in symbol scale.

The reproduced architecture follows the paper's three-stage pipeline:
1. **CNN Backbone:** A ResNet-based feature extractor that captures local, low-level visual textures and stroke features from the input image.
2. **Transformer Encoder:** A Vision Transformer (ViT) infused with **2D Positional Encoding**. This processes the flattened CNN feature maps to capture the complex spatial relationships and global context between multi-line mathematical symbols.
3. **LSTM Decoder with Coverage Attention:** An autoregressive LSTM decoder that generates the target LaTeX sequence. It utilizes a custom Coverage Attention mechanism that maintains a history of past alignments to prevent the network from repeatedly parsing the same symbols (over-parsing) or skipping symbols entirely (under-parsing). 

*Note on Training:* In strict adherence to the paper's explicitly stated methodology, this model was trained using **100% Teacher Forcing** for the entirety of the 300,000 iterations, with checkpoints optimally saved based on validation loss. 

## Repository Structure
```
HVT/
├── backbone.py      # ResNet-based CNN backbone for visual feature extraction
├── encoder.py       # HVT encoder with 2D positional encoding and flattening
├── decoder.py       # LSTM-based decoder step and coverage attention layer
├── model.py         # Main HVTSeq2Seq model orchestrating encoder and decoder
├── dataset.py       # LaTeX tokenization, dataset loader, and batch collation
├── train.py         # Model training script with gradient accumulation and WandB
└── evaluate.py      # Model evaluation script using beam search and BLEU scoring
```

## Installation & Setup

Follow these steps sequentially to set up `uv` (a fast Python package installer and manager), clone the repository, download/unzip the dataset, and prepare the environment.

### 1. Install `uv`
If you do not have `uv` installed, run the installation script:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure environment path (or restart your terminal)
source $HOME/.local/bin/env
```

Verify the installation by running:
```bash
uv --version
```

### 2. Download Project and Dataset
Run these commands sequentially to clone the repository, download the **im2latex-100k** dataset using `curl`, extract it, and clean up:

```bash
# Clone the repository
git clone https://github.com/MAli7319/HVT.git

# Navigate into the project folder
cd HVT

# Download the dataset directly via curl
curl -L -o im2latex100k.zip https://www.kaggle.com/api/v1/datasets/download/shahrukhkhan/im2latex100k

# Extract the dataset into the data folder
unzip im2latex100k.zip -d data/

# Remove the downloaded zip file
rm im2latex100k.zip
```

---

## Running the Model

With `uv` set up, all dependencies are automatically managed and resolved when running script commands.

### Training the Model
To start training the HVT model on the dataset with WandB logging:
```bash
uv run train.py
```

### Evaluating the Model
To evaluate the trained model using batched beam search and calculate the BLEU-4 score:
```bash
uv run evaluate.py
```

## Reproduced Result
The primary objective of this reproduction was to validate the model's structural generation accuracy on the standard **im2latex-100k** dataset. Consistent with the benchmarks established in the original paper, the core evaluation metric is the **Corpus BLEU-4** score evaluated via Beam Search.

### Results Comparison
Below is the comparison between the original results reported by the authors (Table 1) and the highest results achieved in this reproduction on the im2latex-100k test set.

| Model / Methodology | Evaluation Metric | Reported Score |
| :--- | :---: | :---: |
| **Original HVT** (Le et al., Table 1) | BLEU-4 | **89.94%** |
| **Reproduction** | BLEU-4 | **87.09%** |