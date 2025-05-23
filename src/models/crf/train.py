from .crf import *

def save_checkpoint(logger, model, optimizer, epoch, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at {save_path}")

def load_checkpoint(logger, model, optimizer, save_path):
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
    args = {**DEFAULT_ARGS, **args}
    logger = get_logger('crf')
    logger.info(f"Training args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRFModel(num_classes=4, feature_dim=64, input_size=(256,256)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    save_path = os.path.join(save_dir, get_save_name('crf', "pt"))
    start_epoch = 0
    stopping_step = 0
    best_val_loss = float('inf')

    if args["checkpoint_path"]:
        start_epoch = load_checkpoint(logger, model, optimizer, args["checkpoint_path"])
        save_path = args["checkpoint_path"]

    # Train/Val split
    train_len = int(len(dataset)*args["split"][0])
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    trainloader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True)
    valloader = DataLoader(val_set, batch_size=args["batch_size"])

    logger.info(f"{model.__class__.__name__} start training...")

    for epoch in range(start_epoch, args["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}: "):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            final_logits, _ = model(imgs)  # model returns final_logits, emissions
            loss = criterion(final_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(valloader, desc=f"Validating: "):
                imgs, labels = imgs.to(device), labels.to(device)
                final_logits, _ = model(imgs)
                loss = criterion(final_logits, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = val_loss / len(valloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stopping_step = 0
            save_checkpoint(logger, model, optimizer, epoch, save_path)
            print(f"Checkpoint saved at epoch {epoch + 1}.")
        else:
            stopping_step += 1

        logger.info(f"Epoch [{epoch+1}/{args['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if stopping_step > args["patience"]:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"Training finished.")
