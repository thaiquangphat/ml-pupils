from .genetic_algorithm import *
from torch.utils.data import DataLoader, random_split
from utils.logger import get_logger

def train(dataset, save_dir, args):
    args = {**DEFAULT_ARGS, **args}
    logger = get_logger('genetic_algorithm')
    
    criterion = nn.NLLLoss()
    
    batch_size = args['batch_size']
    train_set, test_set = random_split(dataset, [args['split'][0], args['split'][1]])
    train_set, val_set = random_split(dataset, [args["split"][0],args["split"][1]])
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    ga_optimizer = GAOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        result_path=save_dir
    )

    population_metrics_history = ga_optimizer.run()

    logger.info("\nPopulation Metrics History:")
    for gen_index, gen_metrics in enumerate(population_metrics_history, start=1):
        logger.info(f"\nGeneration {gen_index}:")
        for ind_index, metric in enumerate(gen_metrics, start=1):
            logger.info(f"  Individual {ind_index}:")
            logger.info(f"    Fitness: {metric['fitness']:.4f}")
            logger.info(f"    Final Test Accuracy: {metric['test_accuracy']:.4f}")
            logger.info(f"    Training Losses per Epoch: {metric['train_losses']}")
            logger.info(f"    Validation Accuracies per Epoch: {metric['val_accuracies']}")