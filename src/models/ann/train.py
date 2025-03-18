from .ann import *

def train(dataset, save_dir, args): 
    """Train model with checkpointing"""
    args = {**DEFAULT_ARGS, **args}
    
    logger = get_logger('ann')
    logger.info(f'{args}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(f"{args['model']}()").to(device)
    # if args['model'] == 'ResNet18':
    #     model = ResNet18().to(device)
    # else:
    #     model = ANN().to(device)
    
    # model = ANN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    save_path = save_dir / get_save_name(args['model'], "pt")
    start_epoch = 0
    
    if args["checkpoint_path"]:
        start_epoch = load_checkpoint(logger, model, optimizer, args["checkpoint_path"])
        save_path = args["checkpoint_path"]
    
    train_set, val_set = random_split(dataset, [args["split"][0],args["split"][1]])
    trainloader = DataLoader(train_set, batch_size=args["batch_size"])
    valloader = DataLoader(val_set, batch_size=args["batch_size"])

    print(f"{model.__class__.__name__} start training...")
    
    best_val_loss = 1e10
    for epoch in range(start_epoch, args["epochs"]):
        running_loss = 0
        val_loss = 0
        
        model.train()
        for img, label in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            # img = img.unsqueeze(1)
            optimizer.zero_grad()
            output = model.forward(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        for img,label in tqdm(valloader):
            img, label = img.to(device), label.to(device)
            # img = img.unsqueeze(1)
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

    print(f"{model.__class__.__name__} done training")