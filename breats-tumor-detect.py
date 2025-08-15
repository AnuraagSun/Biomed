from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import time
import gc

# Free up memory before starting
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Smaller, more efficient ViT model
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=384, img_size=160):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1):
        super().__init__()
        # Pre-norm architecture (better for training stability)
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),  # Reduced from 4x to 2x multiplier
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, n_classes, emb_size=384, depth=3, heads=6, dropout=0.1, img_size=160):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size=emb_size, img_size=img_size)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_size, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))

# === CONFIG ===
body_part = "breasts"
dataset_path = f"/kaggle/input/filesimages/New folder/Breast"
working_dir = "/kaggle/working"
save_model_as = f"{body_part}_vit.pth"
full_save_path = os.path.join(working_dir, save_model_as)

# Hyperparameters for faster training
img_size = 160          # Smaller image size
batch_size = 32         # Larger batch size if memory allows
emb_size = 384          # Smaller embedding size
depth = 3               # Fewer transformer blocks
heads = 6               # Fewer attention heads
num_workers = 2         # Parallel data loading
epochs = 3              # Keep same number of epochs

print(f"Files in working directory before training: {os.listdir(working_dir)}")

# More efficient transform pipeline
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),  # Simple data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset path {dataset_path} does not exist. Checking similar paths...")
    parent_dir = os.path.dirname(dataset_path)
    if os.path.exists(parent_dir):
        similar_dirs = [d for d in os.listdir(parent_dir) if "breast" in d.lower()]
        if similar_dirs:
            print(f"Found similar directories: {similar_dirs}")
            dataset_path = os.path.join(parent_dir, similar_dirs[0])
            print(f"Using {dataset_path} instead")
    else:
        print(f"Parent directory {parent_dir} not found")

try:
    # Load dataset
    data = datasets.ImageFolder(dataset_path, transform=transform)
    print(f"Successfully loaded dataset with {len(data)} images in {len(data.classes)} classes")
    
    # More efficient data loading
    loader = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create smaller model
    model = ViT(
        n_classes=2, 
        emb_size=emb_size, 
        depth=depth, 
        heads=heads,
        img_size=img_size
    ).to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters (vs ~86M in original)")

    # Better optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=epochs,
        steps_per_epoch=len(loader)
    )

    # === TRAIN ===
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        loss_total = 0
        batch_count = 0
        epoch_start = time.time()
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            out = model(x)
            loss = criterion(out, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track stats
            loss_total += loss.item()
            batch_count += 1
            
            # Print progress and clear memory
            if batch_count % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}, Batch {batch_count}/{len(loader)}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")
                # Clear memory
                if device.type == 'cpu':
                    gc.collect()
                
        # Epoch summary
        epoch_loss = loss_total / len(loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save checkpoint to avoid losing progress
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(working_dir, f"{body_part}_checkpoint.pth"))
    
    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f} seconds")

    # Calculate approximate time saved
    original_time_per_batch = 45 * 60 / 35  # 45 minutes for 35 batches
    new_time_per_batch = total_training_time / (batch_count * epochs)
    speedup = original_time_per_batch / new_time_per_batch
    print(f"Training is approximately {speedup:.1f}x faster")

    # Save final model
    try:
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        torch.save(model.state_dict(), full_save_path)
        
        if os.path.exists(full_save_path):
            file_size_mb = os.path.getsize(full_save_path) / (1024 * 1024)
            print(f"Successfully saved model to {full_save_path} ({file_size_mb:.2f} MB)")
            print(f"Files in working directory after saving: {os.listdir(working_dir)}")
            
            print("\n=== HOW TO DOWNLOAD THE MODEL FILE ===")
            print("1. Look for the 'Output' tab in the right panel of your Kaggle notebook")
            print("2. Find and click on the file named:", save_model_as)
            print("3. Click the download icon to save it to your local machine")
            print("4. Alternatively, run this code for a download link:")
            print("from IPython.display import FileLink")
            print(f"FileLink(r'{full_save_path}')")
        else:
            print(f"ERROR: File was not created at {full_save_path}")
    except Exception as e:
        print(f"ERROR saving model: {e}")

except Exception as e:
    print(f"ERROR during training: {e}")
