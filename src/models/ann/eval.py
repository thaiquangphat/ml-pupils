from .ann import *

def evaluate(dataset, saved_path, args):
    """Evaluate the model using saved checkpoint"""
    args = {**DEFAULT_ARGS, **args}
    
    model = eval(f"{args['model']}()")
    # if 'model' in args and args['model'] == 'resnet18':
    #     model = ResNet18()
    # else:
    #     model = ANN()
        
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            # img = img.unsqueeze(1)
            
            outputs = model.forward(img)
            scores = softmax(outputs,dim=1)
            preds = torch.argmax(scores, dim=1)
            
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return all_labels, all_preds, all_scores
