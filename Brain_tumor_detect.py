import os
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Memory management
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=512, img_size=160):
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
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),  # Reduced from 4x to save memory
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, n_classes, emb_size=512, depth=4, heads=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size=emb_size, img_size=160)
        self.blocks = nn.ModuleList([TransformerBlock(emb_size, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

# === CONFIG ===
body_part = "brain"
dataset_path = "/kaggle/input/filesimages/New folder/Brain"
save_dir = "/kaggle/working"
save_model_as = f"{body_part}_vit.pth"
checkpoint_path = os.path.join(save_dir, f"{body_part}_checkpoint.pth")

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Data transforms with augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Smaller image size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean/std
])

# Load dataset with error handling
try:
    print(f"Loading dataset from {dataset_path}")
    data = datasets.ImageFolder(dataset_path, transform=transform)
    print(f"Found {len(data)} images in {len(data.classes)} classes")
    for i, class_name in enumerate(data.classes):
        count = sum(1 for _, label in data.samples if label == i)
        print(f"  Class {class_name}: {count} images")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Split into train/val
val_size = int(0.2 * len(data))
train_size = len(data) - val_size
train_data, val_data = random_split(data, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))
print(f"Training on {train_size} images, validating on {val_size} images")

# Memory-efficient data loading
batch_size = 8  # Smaller batch size
accum_steps = 4  # Gradient accumulation to simulate larger batches
num_workers = min(2, os.cpu_count() or 1)  # Limited workers to avoid memory issues

train_loader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False
)

val_loader = DataLoader(
    val_data, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViT(
    n_classes=len(data.classes),  # Should be 4 based on directory structure
    emb_size=512,  # Reduced from 768
    depth=4,       # Reduced from 6
    heads=8,
    dropout=0.1
).to(device)

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# Load checkpoint if exists
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

# === TRAIN ===
num_epochs = 10  # More epochs with early stopping
patience = 5     # Early stopping patience
no_improve = 0

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    # Reset gradients at start of epoch
    optimizer.zero_grad()
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch_idx, (images, targets) in enumerate(train_pbar):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets) / accum_steps  # Scale loss for accumulation
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Step optimizer after accumulation steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        train_loss += loss.item() * accum_steps  # Unnormalize loss for reporting
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        train_pbar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * train_correct / train_total
        })
        
        # Free memory periodically
        if (batch_idx + 1) % 10 == 0:
            gc.collect()
    
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        for batch_idx, (images, targets) in enumerate(val_pbar):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            val_pbar.set_postfix({
                'loss': val_loss / (batch_idx + 1),
                'acc': 100. * val_correct / val_total
            })
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Print epoch summary
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_loss < best_val_loss:
        print(f"Val loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, save_model_as))
        no_improve = 0
    else:
        no_improve += 1
        print(f"No improvement for {no_improve} epochs")
    
    # Save checkpoint for resumption
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, checkpoint_path)
    
    # Early stopping
    if no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    # Clear memory between epochs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {os.path.join(save_dir, save_model_as)}")
