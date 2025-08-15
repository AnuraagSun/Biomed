import os
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Memory management
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=512, img_size=192):
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
            nn.Linear(emb_size, emb_size * 2),  # Reduced from 4x to 2x to save memory
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
        self.patch_embed = PatchEmbedding(emb_size=emb_size, img_size=192)
        self.blocks = nn.ModuleList([TransformerBlock(emb_size, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

# === CONFIG ===
body_part = "lungs"
dataset_path = "/kaggle/input/filesimages/New folder/Lung"
save_dir = "/kaggle/working"
save_model_as = f"{body_part}_vit.pth"
checkpoint_path = os.path.join(save_dir, f"{body_part}_vit_checkpoint.pth")

# Smaller image size, better augmentation
transform = transforms.Compose([
    transforms.Resize((192, 192)),  # Reduced from 224 to save memory
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load dataset
print(f"Loading dataset from {dataset_path}")
try:
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes")
    for i, class_name in enumerate(full_dataset.classes):
        class_count = sum(1 for _, label in full_dataset.samples if label == i)
        print(f"Class {class_name}: {class_count} images")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Split dataset into train/validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                         generator=torch.Generator().manual_seed(42))

# Memory-efficient data loading
batch_size = 8  # Reduced from 16 to save memory
num_workers = min(4, os.cpu_count() or 1)  # Limit number of workers
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=False)

# Smaller, more efficient model for CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ViT(n_classes=len(full_dataset.classes), 
           emb_size=512,  # Reduced from 768
           depth=4,       # Reduced from 6
           heads=8, 
           dropout=0.1).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # AdamW with weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Load checkpoint if exists
os.makedirs(save_dir, exist_ok=True)
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from epoch {start_epoch}")

# === TRAIN ===
num_epochs = 10  # Increased from 3 for better accuracy
accum_steps = 4  # Gradient accumulation (simulate larger batch size)
patience = 5     # Early stopping patience
no_improve = 0

for epoch in range(start_epoch, num_epochs):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()  # Zero gradients before accumulation
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps  # Normalize loss for accumulation
        
        # Backward pass
        loss.backward()
        
        # Step optimization after accumulation
        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Track total loss (unnormalized for reporting)
        train_loss += loss.item() * accum_steps
        
        # Memory management
        if (i + 1) % 10 == 0:
            gc.collect()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Update learning rate
    scheduler.step(avg_val_loss)
    
    # Print results
    time_taken = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Time: {time_taken:.1f}s")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
        best_val_loss = avg_val_loss
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
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    
    # Early stopping
    if no_improve >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break
    
    # Memory cleanup
    gc.collect()

print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
print(f"Final model saved to {os.path.join(save_dir, save_model_as)}")
