from .crf import *

def evaluate(dataset, saved_path, args):
    args = {**DEFAULT_ARGS, **args}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRFModel(num_classes=4, feature_dim=64, input_size=(256,256))
    model.to(device)

    if saved_path and os.path.exists(saved_path):
        checkpoint = torch.load(saved_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from {saved_path}")
    else:
        print("No checkpoint found. Using randomly initialized model.")

    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            final_logits, _ = model(imgs)  # only use final_logits
            probs = torch.softmax(final_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_scores
