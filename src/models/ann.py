import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from utils.utils import get_save_name

# default arguments necessary for training ANN
DEFAULT_ARGS = {
    "epochs": 10,
    "log_step": 1,
    "split": [0.9,0.1],
    "checkpoint_path": None,
    "patience": 10,
    "batch_size": 64
}

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

def train(dataset, save_dir, args): 
    """Train model with checkpointing"""
    args = {**DEFAULT_ARGS, **args}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    save_path = save_dir / get_save_name("ann", "pt")
    start_epoch = 0
    
    if args["checkpoint_path"]:
        start_epoch = load_checkpoint(model, optimizer, args["checkpoint_path"])
        save_path = args["checkpoint_path"]
    
    train_set, val_set = random_split(dataset, [args["split"][0],args["split"][1]])
    trainloader = DataLoader(train_set, batch_size=args["batch_size"])
    valloader = DataLoader(val_set, batch_size=args["batch_size"])

    print("ANN start training...")
    
    for epoch in range(start_epoch, args["epochs"]):
        running_loss = 0
        val_loss = 0
        best_val_loss = 1e10
        
        model.train()
        for img, label in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            img = img.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model.forward(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        for img,label in tqdm(valloader):
            img, label = img.to(device), label.to(device)
            img = img.unsqueeze(1)
            output = model.forward(img)
            loss = criterion(output,label)
            
            val_loss += loss.item()
            
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                stopping_step = 0
                save_checkpoint(model, optimizer, epoch, save_path)
            else:
                stopping_step += 1
            
            if stopping_step > args["patience"]:
                print(f"Model stop training at epoch {epoch+1}. Val loss: {val_loss/len(valloader):.4f}")
                break
            
        print(f"Epoch [{epoch+1}/{args["epochs"]}] - Train loss: {running_loss/len(trainloader):.4f} - Val loss: {val_loss/len(valloader):.4f}")

    print(f"ANN done training")



def evaluate(dataset, saved_path, args):
    """Evaluate the model using saved checkpoint"""
    args = {**DEFAULT_ARGS, **args}
    
    model = ANN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if saved_path:
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from {saved_path}")
    else:
        print("No checkpoint found, evaluating with randomly initialized model.")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    dataloader = DataLoader(dataset,batch_size=args["batch_size"])
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
