import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import MathTokenizer, Im2LatexDataset
from model import HVTSeq2Seq

def decode_sequence(token_ids, tokenizer):
    words = []
    for idx in token_ids:
        word = tokenizer.idx2word.get(idx.item(), tokenizer.unk_token)
        if word == tokenizer.eos_token:
            break
        if word not in [tokenizer.sos_token, tokenizer.pad_token]:
            words.append(word)
    return words

def main():
    # Update this to your 'hvt_model_best.pth'
    checkpoint_path = "hvt_model_best.pth" 
    data_folder = 'data'
    max_seq_len = 150
    batch_size = 16 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    formulas_path = os.path.join(data_folder, 'im2latex_formulas.norm.csv')
    with open(formulas_path, 'r', encoding='utf-8') as f:
        all_formulas = [line.strip() for line in f.readlines()]

    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_formulas)
    
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

    model = HVTSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    references = []  
    hypotheses = []  
    
    sos_idx = tokenizer.word2idx[tokenizer.sos_token]
    eos_idx = tokenizer.word2idx[tokenizer.eos_token]
    pad_idx = tokenizer.word2idx[tokenizer.pad_token]

    print("Starting Beam Search evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, target_seqs) in enumerate(test_loader):
            images = images.to(device)
            B = images.size(0)
            
            # --- FIX: Trigger the dedicated Beam Search Generator ---
            pred_ids = model.generate(images, sos_idx, eos_idx, pad_idx, beam_size=5)
            
            for i in range(B):
                pred_tokens = decode_sequence(pred_ids[i], tokenizer)
                true_tokens = decode_sequence(target_seqs[i], tokenizer)
                
                hypotheses.append(pred_tokens)
                references.append([true_tokens]) 
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches...")

    print("\nCalculating Corpus BLEU-4 Score...")
    chencherry = SmoothingFunction()
    bleu_score = corpus_bleu(
        references, 
        hypotheses, 
        weights=(0.25, 0.25, 0.25, 0.25), 
        smoothing_function=chencherry.method1
    )
    
    print("\n--- Visual Sanity Check ---")
    print(f"Ground Truth : {' '.join(references[0][0])}")
    print(f"Model Predict: {' '.join(hypotheses[0])}")
    
    print(f"\n====================================")
    print(f"Final Test BLEU-4 Score: {bleu_score * 100:.2f}")
    print(f"====================================")

if __name__ == "__main__":
    main()