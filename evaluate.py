import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import MathTokenizer, Im2LatexDataset
from model import HVTSeq2Seq

def decode_sequence(token_ids, tokenizer):
    """Converts a list of integer IDs back into a space-separated string, stripping special tokens."""
    words = []
    for idx in token_ids:
        word = tokenizer.idx2word.get(idx.item(), tokenizer.unk_token)
        if word == tokenizer.eos_token:
            break
        if word not in [tokenizer.sos_token, tokenizer.pad_token]:
            words.append(word)
    return words

def main():
    # --- 1. Setup Configuration ---
    # Change this to point to your best saved epoch!
    checkpoint_path = "hvt_model_checkpoint_10.pth" 
    data_folder = 'data'
    max_seq_len = 150
    batch_size = 16 # Evaluation uses less VRAM, we can safely bump this up

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # --- 2. Build Tokenizer ---
    formulas_path = os.path.join(data_folder, 'im2latex_formulas.norm.csv')
    with open(formulas_path, 'r', encoding='utf-8') as f:
        all_formulas = [line.strip() for line in f.readlines()]

    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_formulas)
    print(f"Vocabulary loaded. Total unique tokens: {tokenizer.vocab_size}")

    # --- 3. Load the Test Dataset ---
    # We do NOT use RandomRotation or RandomAffine during evaluation!
    test_transforms = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = Im2LatexDataset(
        data_dir=data_folder,
        split_csv='im2latex_test.csv',
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        transform=test_transforms
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 4. Load the Trained Model ---
    model = HVTSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {checkpoint_path}. Train the model first!")
        return

    # Put model in eval mode (disables Dropout and freezes BatchNorm)
    model.eval()

    # --- 5. The Inference Loop ---
    references = []  # Ground truth lists
    hypotheses = []  # Model prediction lists
    
    # We need a dummy target sequence filled with <SOS> tokens to kickstart the decoder
    sos_idx = tokenizer.word2idx[tokenizer.sos_token]

    print("Starting evaluation...")
    
    # Disable gradient tracking to save massive amounts of VRAM and speed up inference
    with torch.no_grad():
        for batch_idx, (images, target_seqs) in enumerate(test_loader):
            images = images.to(device)
            B = images.size(0)
            
            # Create a dummy target tensor where the first token is <SOS>
            dummy_targets = torch.zeros((B, max_seq_len), dtype=torch.long).to(device)
            dummy_targets[:, 0] = sos_idx
            
            # Forward pass with 0% Teacher Forcing (pure autoregressive generation)
            predictions = model(images, target_seqs=dummy_targets, teacher_forcing_ratio=0.0)
            
            # Get the highest probability token at each time step (Greedy Search)
            pred_ids = predictions.argmax(dim=-1)
            
            # Convert tensors back to lists of string tokens
            for i in range(B):
                pred_tokens = decode_sequence(pred_ids[i], tokenizer)
                true_tokens = decode_sequence(target_seqs[i], tokenizer)
                
                hypotheses.append(pred_tokens)
                # BLEU expects a list of references for each hypothesis
                references.append([true_tokens]) 
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # --- 6. Calculate BLEU-4 Score ---
    print("\nCalculating Corpus BLEU-4 Score...")
    
    # We use a smoothing function to prevent zero scores if a short formula misses a 4-gram
    chencherry = SmoothingFunction()
    
    bleu_score = corpus_bleu(
        references, 
        hypotheses, 
        weights=(0.25, 0.25, 0.25, 0.25), 
        smoothing_function=chencherry.method1
    )
    
    # Print a random example to visually verify
    print("\n--- Visual Sanity Check ---")
    print(f"Ground Truth : {' '.join(references[0][0])}")
    print(f"Model Predict: {' '.join(hypotheses[0])}")
    
    # BLEU is usually reported as a percentage (0 to 100)
    print(f"\n====================================")
    print(f"Final Test BLEU-4 Score: {bleu_score * 100:.2f}")
    print(f"====================================")

if __name__ == "__main__":
    main()