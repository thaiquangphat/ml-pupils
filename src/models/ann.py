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
import torch
import torch.nn as nn

# basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  

    def forward(self, x):
        identity = x  # Save input for the skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Add the skip connection
        out = self.relu(out)  # Apply ReLU after addition

        return out

# Define ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=4): 
        super(ResNet18, self).__init__()
        self.in_channels = 64  # Initial number of channels after the first conv layer

        # Initial Convolution + BatchNorm + ReLU + MaxPool
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 2, stride=1)   
        self.layer2 = self._make_layer(128, 2, stride=2) 
        self.layer3 = self._make_layer(256, 2, stride=2) 
        self.layer4 = self._make_layer(512, 2, stride=2) 

        # Global Average Pooling + Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        """
        Creates a ResNet layer consisting of residual blocks.
        - out_channels: Number of output channels
        - blocks: Number of residual blocks
        - stride: Stride for the first block
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion 
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

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
    model = ResNet18().to(device)
    # model = ANN().to(device)
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
            
        logger.info(f"Epoch [{epoch+1}/{args['epochs']}] - Train loss: {running_loss/len(trainloader):.4f} - Val loss: {val_loss/len(valloader):.4f}")

    print(f"ANN done training")



def evaluate(dataset, saved_path, args):
    """Evaluate the model using saved checkpoint"""
    args = {**DEFAULT_ARGS, **args}
    
    model = ResNet18()
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
