# HVT (Hybrid Vision Transformer) for Image-to-LaTeX

This repository contains a deep learning model designed for converting mathematical formula images into their corresponding LaTeX sequence representations. The architecture utilizes a hybrid approach: a Convolutional Neural Network (CNN) feature extractor, a Transformer-based Encoder with 2D Positional Encoding, and an LSTM-based Decoder with Coverage Attention.

This document provides a comprehensive technical overview of every file, class, and method within the refactored codebase.

---

## 1. `backbone.py`
This module defines the CNN feature extractor responsible for converting the raw input image into a 2D grid of high-level features.

### `BasicBlock(nn.Module)`
A standard ResNet residual block structure.
- **`__init__(self, in_channels, out_channels, stride=(1, 1))`**: Initializes two $3 \times 3$ convolutional layers, each followed by Batch Normalization. If `stride` is not `1` or input channels do not match output channels, a `downsample` layer (a $1 \times 1$ Conv + BatchNorm) is created to match dimensions for the residual connection.
- **`forward(self, x)`**: Applies the first convolution, batch norm, and ReLU, followed by the second convolution and batch norm. Finally, it adds the residual `identity` (the original input `x`, optionally downsampled) to the output before applying a final ReLU activation.

### `MathResNetBackbone(nn.Module)`
A custom ResNet-like backbone specifically tailored to process formula images.
- **`__init__(self)`**: Starts with a $7 \times 7$ Convolution (`stride=2`, `padding=3`) mapping 1-channel grayscale images to 64 channels. This is followed by a $3 \times 3$ MaxPool. Then, it defines 4 sequential ResNet layers using `_make_layer()`. 
  - Layer 1: 64 channels, 2 blocks.
  - Layer 2: 128 channels, 2 blocks, `stride=(2, 2)` (downsamples spatially).
  - Layer 3: 256 channels, 2 blocks, `stride=(1, 2)` (downsamples width more than height, useful for long formulas).
  - Layer 4: 512 channels, 2 blocks, `stride=(1, 2)`.
- **`_make_layer(self, out_channels, blocks, stride)`**: Helper function to sequentially chain a downsampling `BasicBlock` followed by standard `BasicBlock`s.
- **`forward(self, x)`**: Passes the input tensor `x` sequentially through the initial convolution, max pool, and the four ResNet layers.

---

## 2. `encoder.py`
This module transforms the 2D feature map from the backbone into a sequence of context-aware, globally attended feature vectors.

### `PositionalEncoding2D(nn.Module)`
Injects spatial awareness into the features, since Transformers inherently lack a concept of 2D geometry.
- **`__init__(self, d_model=512, max_h=128, max_w=512)`**: Generates continuous sinusoidal positional encodings. It splits `d_model` in half; one half encodes the Y (height) position, and the other half encodes the X (width) position using sine for even indices and cosine for odd indices.
- **`forward(self, x)`**: Slices the pre-computed positional encoding grid to match the actual spatial dimensions (H, W) of the input `x` and adds the encoding directly to `x`.

### `FeatureFlattening(nn.Module)`
- **`forward(self, x)`**: Flattens the spatial dimensions (Height and Width) of the feature map into a single sequence dimension, preparing it for the Transformer. The resulting tensor is permuted from `[Batch, Channels, H*W]` to `[Batch, H*W, Channels]`.

### `CustomEncoderLayer(nn.Module)`
A single Transformer Encoder block implementing a "Pre-Norm" architecture, which provides better gradient flow and stability.
- **`__init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)`**: Initializes `MultiheadAttention`, two linear layers for the FeedForward network, and two `LayerNorm` instances.
- **`forward(self, src, src_mask=None, src_key_padding_mask=None)`**: Applies LayerNorm *before* passing the input into the Self-Attention mechanism (Block 1) and FeedForward network (Block 2). Residual connections are added to the un-normed input in both blocks.

### `HVTEncoder(nn.Module)`
The master wrapper for the encoding stage.
- **`__init__(self, backbone, d_model=512, nhead=8, num_layers=4)`**: Takes the CNN `backbone`, initializes the 2D Positional Encoder, Feature Flattener, and the stacked `TransformerEncoder`. Critically, it defines a learnable `[CLS]` (classification) token parameter.
- **`forward(self, images)`**: 
  1. Extracts features via the CNN backbone.
  2. Adds 2D Positional Encoding.
  3. Flattens features into a 1D sequence.
  4. Prepends the learnable `[CLS]` token to the sequence for every item in the batch.
  5. Passes the combined sequence through the stacked Transformer layers.

---

## 3. `decoder.py`
This module contains the Recurrent sequence-generation logic to predict LaTeX tokens one-by-one.

### `CoverageAttention(nn.Module)`
An attention mechanism that keeps track of its past attention history ("coverage") to penalize the model for repeatedly focusing on the same spatial locations in the image (preventing token repetition loops).
- **`__init__(self, encoder_dim, decoder_dim, attention_dim)`**: Initializes linear projections for the encoder outputs (`W_enc`) and decoder hidden state (`W_dec`). It also initializes a 1D Convolution (`W_cov`) to process the past coverage vector.
- **`forward(self, hidden, encoder_outputs, coverage)`**: 
  1. Projects encoder states, hidden states, and the past coverage vector into a shared `attention_dim`.
  2. Applies `tanh` and a linear projection `V` to compute raw attention scores, followed by `softmax` to yield `attention_weights`.
  3. Updates the `coverage` by accumulating the current `attention_weights`.
  4. Calculates the `context_vector` as the weighted sum of `encoder_outputs`.

### `HVTDecoderStep(nn.Module)`
Represents a single timestep execution of the LSTM sequence generator.
- **`__init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim)`**: Defines an `Embedding` layer for the target vocabulary, initializes the `CoverageAttention` module, sets up an `LSTMCell`, and creates the final `Linear` classifier to project features back to `vocab_size`.
- **`forward(self, prev_token, hidden, cell, encoder_outputs, coverage)`**: 
  1. Embeds the previously generated token (or `<SOS>`).
  2. Computes the `context` vector and updated `coverage` via the Attention mechanism.
  3. Concatenates the embedded token and `context`, passing it into the `LSTMCell` to yield `next_hidden` and `next_cell`.
  4. Concatenates `next_hidden` and `context` and passes them through the classifier to produce `prediction_logits` over the vocabulary.

---

## 4. `model.py`
This file ties the Encoder and Decoder together into the end-to-end framework.

### `HVTSeq2Seq(nn.Module)`
- **`__init__(self, vocab_size, d_model=512, decoder_dim=256, max_seq_len=150)`**: Initializes the `MathResNetBackbone`, `HVTEncoder`, and `HVTDecoderStep`. It also defines two linear projections (`init_h`, `init_c`) used to transform the Encoder's final `[CLS]` token into the initial hidden and cell states for the Decoder LSTM.
- **`forward(self, images, target_seqs=None, teacher_forcing_ratio=0.5)`**: 
  1. Passes the image through the Encoder.
  2. Splits the encoder output: The first vector (the `[CLS]` token) is passed through `init_h`/`init_c` to initialize the LSTM states. The remaining vectors (`patch_outputs`) serve as the spatial features to attend over.
  3. Initializes empty tracking tensors for outputs and coverage.
  4. **Generation Loop**: Iterates up to `max_seq_len`. In each step, it calls `self.decoder_step`.
  5. **Teacher Forcing**: During training (if `target_seqs` is provided), the model will randomly decide (based on `teacher_forcing_ratio`) whether to feed the *true* target token to the next step, or its own *predicted* top-1 token.

---

## 5. `dataset.py`
This module manages loading, formatting, and converting the raw formula string data and images into PyTorch tensors.

### `MathTokenizer`
Handles mapping human-readable string tokens to machine-readable integer indices.
- **`__init__(self)`**: Pre-defines special tokens: `<PAD>` (0), `<SOS>` (1, Start of Sequence), `<EOS>` (2, End of Sequence), and `<UNK>` (3, Unknown).
- **`build_vocab(self, formulas)`**: Scans a list of space-separated formula strings. Any previously unseen token is added to the vocabulary mappings (`word2idx`, `idx2word`), dynamically expanding `vocab_size`.
- **`encode(self, formula, max_length)`**: Converts a single string formula to a list of IDs. It bounds the sequence with `<SOS>` and `<EOS>`, replaces unseen tokens with `<UNK>`, truncates if the length exceeds `max_length`, and pads with `<PAD>` if it falls short.

### `Im2LatexDataset(Dataset)`
A standard PyTorch dataset implementation.
- **`__init__(self, data_dir, split_csv, tokenizer, max_seq_len, transform)`**: Reads a CSV file using Pandas. The parser is carefully configured (`quotechar='"'`) to handle commas correctly embedded inside LaTeX strings.
- **`__getitem__(self, idx)`**: Fetches a single row.
  1. Resolves the image path and opens it as Grayscale (`'L'`). If missing, it creates a blank fallback image to prevent crashes.
  2. Applies torchvision `transform`s.
  3. Passes the raw formula string to the `tokenizer.encode()` method to get the `target_seq` tensor.
  4. Returns the `(image, target_seq)` tuple.

---

## 6. `train.py`
The top-level execution script for training the `HVTSeq2Seq` model.

### `main()`
- **Transforms**: Configures `torchvision.transforms` to rigidly resize all images to `128x512` (to match the hard-coded max dimensions in the `PositionalEncoding2D`), convert to Tensors, and normalize them to a `[-1, 1]` range (`mean=[0.5], std=[0.5]`).
- **Initialization**: Loads raw formulas to build the `MathTokenizer` vocabulary. Initializes the `Im2LatexDataset` and wraps it in a PyTorch `DataLoader` with batching and shuffling.
- **Optimization**: Sets up the model on the available device (`cuda` or `cpu`). Initializes the `Adam` optimizer (learning rate $1e^{-4}$) and a `CrossEntropyLoss` function specifically configured to ignore `<PAD>` tokens so the model isn't penalized for padding predictions.
- **Training Loop**: 
  - Iterates for a specified number of epochs (default 10).
  - Dynamically calculates `forcing_ratio`, decaying it linearly from $1.0$ (100% teacher forcing) down to $0.5$ as epochs progress, forcing the model to rely more on its own predictions over time.
  - Flattens the predictions and targets to calculate the loss.
  - Employs `clip_grad_norm_` (max_norm=1.0) before the optimizer step to prevent exploding gradients, a common issue with LSTMs.
  - Saves the resulting model state dictionary `.pth` to disk at the end of every epoch.
