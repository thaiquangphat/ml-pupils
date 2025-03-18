from .genetic_algorithm import *
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import glob

def evaluate(dataset, saved_path, args):
    test_loader = DataLoader(dataset, batchsize=args['batch_size'], shuffle=False)
    
    saved_dir = src_path / 'results' / 'models' / 'genetic_algorithm' / 'best_model'
    saved_path = glob.glob(str(saved_dir) + '/*.pt') 
   
    
    if os.path.exists(saved_path):
        model = torch.load(saved_path)
        print(f"Loaded model from {saved_path}")
    else:
        raise FileNotFoundError("No GAoptimized model found.")
    
    device = 'cpu'
    model.to(device)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            # img = img.unsqueeze(1)
            
            outputs = model.forward(img)
            scores = softmax(outputs,dim=1)
            preds = torch.argmax(scores, dim=1)
            
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return all_labels, all_preds, all_scores