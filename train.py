import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb 

from dataset import MathTokenizer, Im2LatexDataset, pad_collate_fn
from model import HVTSeq2Seq

""" Running the main training and validation cycle with WandB integration and gradient accumulation. """
def main():
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=5), 
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    val_transforms = transforms.Compose([
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
        transform=train_transforms
    )

    val_dataset = Im2LatexDataset(
        data_dir=data_folder,
        split_csv='im2latex_validate.csv',
        tokenizer=tokenizer,
        max_seq_len=150,
        transform=val_transforms
    )

    micro_batch_size = 32  
    effective_batch_size = 32
    accumulation_steps = effective_batch_size // micro_batch_size
    total_iterations = 300000
    learning_rate = 5e-4
    weight_decay = 2e-6

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

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=4, collate_fn=pad_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HVTSeq2Seq(vocab_size=tokenizer.vocab_size).to(device)

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

    global_step = 0
    epoch = 0
    best_val_loss = float('inf')

    while global_step < total_iterations:
        epoch += 1
        model.train()
        total_train_loss = 0.0 
        batches_processed = 0
        optimizer.zero_grad() 

        """ Looping over training batches, computing logits, calculating loss, and updating parameters. """
        for batch_idx, (images, target_seqs) in enumerate(train_loader):
            if global_step >= total_iterations:
                break

            images = images.to(device)
            target_seqs = target_seqs.to(device)
            batches_processed += 1

            predictions = model(images, target_seqs, teacher_forcing_ratio=1.0)

            pred_flat = predictions[:, 1:, :].reshape(-1, tokenizer.vocab_size)
            target_flat = target_seqs[:, 1:].reshape(-1)

            loss = criterion(pred_flat, target_flat) / accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()

                global_step += 1

                if global_step % 10 == 0:  
                    current_lr = scheduler.get_last_lr()[0]
                    display_loss = loss.item() * accumulation_steps 

                    wandb.log({
                        "train/loss": display_loss,
                        "train/learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": global_step
                    })

                if global_step % 100 == 0: 
                    print(f"Step {global_step}/{total_iterations} | Loss: {display_loss:.4f} | LR: {current_lr:.6f}")

        if batches_processed > 0:
            avg_train_loss = total_train_loss / batches_processed

            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                """ Computing validation loss over the evaluation split without tracking gradients. """
                for val_images, val_target_seqs in val_loader:
                    val_images = val_images.to(device)
                    val_target_seqs = val_target_seqs.to(device)

                    val_preds = model(val_images, val_target_seqs, teacher_forcing_ratio=1.0)

                    val_pred_flat = val_preds[:, 1:, :].reshape(-1, tokenizer.vocab_size)
                    val_target_flat = val_target_seqs[:, 1:].reshape(-1)

                    v_loss = criterion(val_pred_flat, val_target_flat)
                    total_val_loss += v_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            wandb.log({
                "val/loss": avg_val_loss,
                "epoch": epoch,
                "global_step": global_step
            })

            print(f"=== Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")

            if avg_val_loss < best_val_loss:
                print(f"New best validation loss! ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "hvt_model_best.pth")
            else:
                print(f"Validation loss did not improve from {best_val_loss:.4f}.")

        torch.save(model.state_dict(), "hvt_model_latest.pth")

    wandb.finish()

if __name__ == "__main__":
    main()