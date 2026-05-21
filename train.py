import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb # <-- 1. Import W&B

from dataset import MathTokenizer, Im2LatexDataset
from model import HVTSeq2Seq

def main():
    # --- PAPER SPEC 1: Data Augmentation ---
    image_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=5), 
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), 
        transforms.Resize((128, 512)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    data_folder = 'data'
    formulas_path = os.path.join(data_folder, 'im2latex_formulas.norm.csv')

    with open(formulas_path, 'r', encoding='utf-8') as f:
        all_formulas = [line.strip() for line in f.readlines()]

    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_formulas)
    print(f"Vocabulary built! Total unique tokens: {tokenizer.vocab_size}")

    train_dataset = Im2LatexDataset(
        data_dir=data_folder,
        split_csv='im2latex_train.csv',
        tokenizer=tokenizer,
        max_seq_len=150,
        transform=image_transforms
    )

    micro_batch_size = 8  
    effective_batch_size = 32
    accumulation_steps = effective_batch_size // micro_batch_size
    total_iterations = 300000
    learning_rate = 5e-4
    weight_decay = 2e-6

    # --- 2. Initialize W&B Dashboard ---
    wandb.init(
        project="hvt-im2latex",
        name="hvt-resnet-lstm-run1",
        config={
            "architecture": "HVT (ResNet + ViT + LSTM)",
            "dataset": "im2latex-100k",
            "micro_batch_size": micro_batch_size,
            "effective_batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_seq_len": 150,
            "iterations": total_iterations,
            "teacher_forcing": 1.0
        }
    )

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HVTSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)

    # --- 3. Watch the Model Gradients ---
    # Log gradients and parameters every 100 steps to catch vanishing/exploding gradients
    wandb.watch(model, log="all", log_freq=100)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        total_steps=total_iterations,
        pct_start=0.05, 
        anneal_strategy='cos'
    )

    pad_idx = tokenizer.word2idx[tokenizer.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"--- Starting Training for {total_iterations} Iterations ---")
    model.train()
    
    global_step = 0
    epoch = 0
    optimizer.zero_grad() 

    while global_step < total_iterations:
        epoch += 1
        
        for batch_idx, (images, target_seqs) in enumerate(train_loader):
            if global_step >= total_iterations:
                break
                
            images = images.to(device)
            target_seqs = target_seqs.to(device)
            
            predictions = model(images, target_seqs, teacher_forcing_ratio=1.0)
            
            pred_flat = predictions[:, 1:, :].reshape(-1, tokenizer.vocab_size)
            target_flat = target_seqs[:, 1:].reshape(-1)
            
            loss = criterion(pred_flat, target_flat) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                
                global_step += 1
                
                # --- 4. Log Live Metrics to the Dashboard ---
                if global_step % 10 == 0:  # Log to wandb frequently
                    current_lr = scheduler.get_last_lr()[0]
                    display_loss = loss.item() * accumulation_steps 
                    
                    wandb.log({
                        "train/loss": display_loss,
                        "train/learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": global_step
                    })
                
                if global_step % 100 == 0: # Print to terminal occasionally
                    print(f"Step {global_step}/{total_iterations} | Loss: {display_loss:.4f} | LR: {current_lr:.6f}")
                    
        torch.save(model.state_dict(), f"hvt_model_checkpoint_{epoch}.pth")
        print(f"=== Epoch {epoch} Complete | Model Checkpoint Saved ===")
        
    # --- 5. Finish the W&B Run ---
    wandb.finish()

if __name__ == "__main__":
    main()