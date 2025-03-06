import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from utils.utils import get_save_name
from utils.logger import get_logger

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
    """Implement simple ANN model"""
    def __init__(self, num_classes=4):
        super(ANN, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # (256x256) -> (256x256)
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256x256) -> (128x128)
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),  # (128x128) -> (128x128)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128x128) -> (64x64)
            nn.Dropout(0.3),
        )

        self.feature3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (64x64) -> (64x64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64x64) -> (32x32)
            nn.Dropout(0.3),
        )

        self.feature4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32x32) -> (32x32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32x32) -> (16x16)
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 120, kernel_size=3, stride=1, padding=1),  # (16x16) -> (16x16)
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling -> (1x1)
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = x.view(x.shape[0], -1) # flatten each sample
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, dim=1)
    
def save_checkpoint(logger, model, optimizer, epoch, save_path):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at {save_path}")

def load_checkpoint(logger, model, optimizer, save_path):
    """Load model checkpoint if available"""
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    return start_epoch

def train(dataset, save_dir, args): 
    """Train model with checkpointing"""
    args = {**DEFAULT_ARGS, **args}
    
    logger = get_logger('ann')
    logger.info(f'{args}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    save_path = save_dir / get_save_name("ann", "pt")
    start_epoch = 0
    
    if args["checkpoint_path"]:
        start_epoch = load_checkpoint(logger, model, optimizer, args["checkpoint_path"])
        save_path = args["checkpoint_path"]
    
    train_set, val_set = random_split(dataset, [args["split"][0],args["split"][1]])
    trainloader = DataLoader(train_set, batch_size=args["batch_size"])
    valloader = DataLoader(val_set, batch_size=args["batch_size"])

    print("ANN start training...")
    
    best_val_loss = 1e10
    for epoch in range(start_epoch, args["epochs"]):
        running_loss = 0
        val_loss = 0
        
        
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
            
        if val_loss/len(valloader) <= best_val_loss:
            best_val_loss = val_loss/len(valloader)
            stopping_step = 0
            save_checkpoint(logger, model, optimizer, epoch, save_path)
        else:
            stopping_step += 1
        
        if stopping_step > args["patience"]:
            logger.info(f"Model stop training at epoch {epoch+1}. Val loss: {val_loss/len(valloader):.4f}")
            break
            
        logger.info(f"Epoch [{epoch+1}/{args["epochs"]}] - Train loss: {running_loss/len(trainloader):.4f} - Val loss: {val_loss/len(valloader):.4f}")

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
    
    dataloader = DataLoader(dataset,batch_size=args["batch_size"],shuffle=False)
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
