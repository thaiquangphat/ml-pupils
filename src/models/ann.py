import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.utils import get_save_name

class ANN(nn.Module):
    """Implement simple LeNet5 with batch norm"""
    def __init__(self, num_classes=4):
        super(ANN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(6), 
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 62 * 62, 120),
            # nn.BatchNorm1d(120), 
            nn.ReLU(),
            nn.Linear(120, 84),
            # nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1) # flatten each sample
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, dim=1)
    
def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, save_path):
    """Load model checkpoint if available"""
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    return start_epoch

def train(dataloader, save_dir, checkpoint_path, num_epochs=10, log_step=1):
    """Train model with checkpointing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    save_path = save_dir / get_save_name("ann", "pt")
    start_epoch = 0
    
    if checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        save_path = checkpoint_path
    
    print("ANN start training...")
    
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            img = img.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model.forward(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if epoch % log_step == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print(f"ANN done training. Checkpoint saved at {save_path}")

def evaluate(dataloader, save_path):
    """Evaluate the model using saved checkpoint"""
    model = ANN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from {save_path}")
    else:
        print("No checkpoint found, evaluating with randomly initialized model.")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            img = img.unsqueeze(1)
            
            outputs = model.forward(img)
            scores = softmax(outputs,dim=1)
            preds = torch.argmax(scores, dim=1)
            
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return all_labels, all_preds, all_scores
