import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import os
from pathlib import Path

DEFAULT_ARGS = {
    'batch_size': 32,
    'split': [0.9, 0.1]
}

src_path = Path(os.path.dirname(os.path.dirname(os.getcwd())))

class DynamicNN(nn.Module):
    def __init__(self,
                 num_cnns=2,
                 num_fcns=2,
                 img_size=(1,256, 256),
                 filters=[32, 64],
                 kernel_sizes=[3, 3],
                 strides=[1, 1],
                 pool_sizes=[2, 2],
                 fcn_units=[128, 64],
                 activation_index=0,
                 lr=0.01):
        super(DynamicNN, self).__init__()

        self.num_cnns = num_cnns
        self.num_fcns = num_fcns
        self.img_size = img_size
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.fcn_units = fcn_units

        self.activate_function_list = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
        self.activation_index = activation_index
        self.activation_function = self.activate_function_list[self.activation_index]

        self.conv_layers = self._build_conv_layers()
        self.fc_layers = self._build_fc_layers()

        self.lr = lr
        
    def _build_conv_layers(self):
        layers = []
        in_channels = self.img_size[0]

        for i in range(self.num_cnns):
            layers.append(nn.Conv2d(in_channels,
                                    self.filters[i],
                                    kernel_size=self.kernel_sizes[i],
                                    stride=self.strides[i]))
            layers.append(self.activation_function)
            layers.append(nn.MaxPool2d(self.pool_sizes[i]))
            in_channels = self.filters[i]

        return nn.Sequential(*layers)

    def _build_fc_layers(self):
        layers = []
        H, W = self.img_size[1], self.img_size[2]
        for i in range(self.num_cnns):
            H = (H - self.kernel_sizes[i]) // self.strides[i] + 1
            W = (W - self.kernel_sizes[i]) // self.strides[i] + 1
            H = H // self.pool_sizes[i]
            W = W // self.pool_sizes[i]

        in_features = self.filters[-1] * H * W

        for units in self.fcn_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(self.activation_function)
            in_features = units

        layers.append(nn.Linear(self.fcn_units[-1], 4))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


    def __str__(self):
        info = "DynamicNN Architecture:\n"
        info += f" Input Image Size: {self.img_size}\n"
        info += f" Number of CNN Layers: {self.num_cnns}\n"
        for i in range(self.num_cnns):
            in_channels = self.img_size[0] if i == 0 else self.filters[i-1]
            info += f"  CNN Layer {i+1}:\n"
            info += f"    Conv2d(in_channels={in_channels}, out_channels={self.filters[i]}, kernel_size={self.kernel_sizes[i]}, stride={self.strides[i]})\n"
            info += f"    MaxPool2d(pool_size={self.pool_sizes[i]})\n"
        info += f" Number of FCN Layers: {self.num_fcns}\n"
        for i in range(self.num_fcns):
            info += f"  FCN Layer {i+1}: Linear(units={self.fcn_units[i]})\n"
        info += f" Final Classification Layer: Linear({self.fcn_units[-1]} -> 4) + Softmax\n"
        info += f" Activation Function: {self.activation_function.__class__.__name__}\n"
        info += f" Learning Rate: {self.lr}\n"
        return info
    
class GAOptimizer:
    def __init__(self,
                 population_size=10,
                 generations=10,
                 mutation_rate=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 epochs_per_model=20,
                 lr_choices=None,
                 activation_functions=None,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 criterion=None,
                 result_path=src_path / 'results' / 'genetic_algorithm'):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.device = device
        print(f"Training on {device}")
        self.epochs_per_model = epochs_per_model

        if lr_choices is None:
            self.lr_choices = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        else:
            self.lr_choices = lr_choices

        if activation_functions is None:
            self.activation_functions = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
        else:
            self.activation_functions = activation_functions

        self.population = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if criterion is None:
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = criterion

        self.result_path = result_path
        
    def initialize_population(self,
                              num_cnn_layers_range=(1, 4),
                              num_fcn_layers_range=(1, 4)):
        print(f"Initialize population")
        for _ in range(self.population_size):
            cnn_layers = random.randint(*num_cnn_layers_range)
            fcn_layers = random.randint(*num_fcn_layers_range)
            activation_index = random.randint(0, len(self.activation_functions) - 1)
            lr = random.choice(self.lr_choices)

            possible_filters = [16, 32, 64, 128]
            filters = []
            for i in range(cnn_layers):
                if i == 0:
                    filt = random.choice(possible_filters)
                else:
                    allowed = [f for f in possible_filters if f <= filters[i - 1]]
                    filt = random.choice(allowed)
                filters.append(filt)

            allowed_kernel_sizes = [3, 5, 7]
            kernel_sizes = []
            for i in range(cnn_layers):
                if i == 0:
                    k = random.choice(allowed_kernel_sizes)
                else:
                    allowed = [ks for ks in allowed_kernel_sizes if ks <= kernel_sizes[i - 1]]
                    k = random.choice(allowed)
                kernel_sizes.append(k)

            strides = [random.randint(1, 2) for _ in range(cnn_layers)]

            allowed_pool_sizes = [2, 3, 4]
            pool_sizes = []
            for i in range(cnn_layers):
                if i == 0:
                    p = random.choice(allowed_pool_sizes)
                else:
                    allowed = [ps for ps in allowed_pool_sizes if ps <= pool_sizes[i - 1]]
                    p = random.choice(allowed)
                pool_sizes.append(p)

            fcn_units = [random.randint(32, 256) for _ in range(fcn_layers)]

            individual = DynamicNN(num_cnns=cnn_layers,
                                   num_fcns=fcn_layers,
                                   filters=filters,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   pool_sizes=pool_sizes,
                                   fcn_units=fcn_units,
                                   activation_index=activation_index,
                                   lr=lr).to(self.device)
            self.population.append(individual)


    def crossover(self, parent1, parent2):
        child_cnn_layers = random.choice([parent1.num_cnns, parent2.num_cnns])
        child_fcn_layers = random.choice([parent1.num_fcns, parent2.num_fcns])
        child_activation_index = random.choice([parent1.activation_index, parent2.activation_index])
        child_lr = random.choice([parent1.lr, parent2.lr])

        def choose_param(list1, list2, idx):
            if idx < len(list1) and idx < len(list2):
                return random.choice([list1[idx], list2[idx]])
            elif idx < len(list1):
                return list1[idx]
            elif idx < len(list2):
                return list2[idx]
            else:
                return None  

        child_filters = []
        child_kernel_sizes = []
        child_strides = []
        child_pool_sizes = []
        for i in range(child_cnn_layers):
            filt = choose_param(parent1.filters, parent2.filters, i)
            kernel = choose_param(parent1.kernel_sizes, parent2.kernel_sizes, i)
            stride = choose_param(parent1.strides, parent2.strides, i)
            pool = choose_param(parent1.pool_sizes, parent2.pool_sizes, i)
            child_filters.append(filt)
            child_kernel_sizes.append(kernel)
            child_strides.append(stride)
            child_pool_sizes.append(pool)

        for i in range(1, child_cnn_layers):
            if child_filters[i] > child_filters[i - 1]:
                child_filters[i] = child_filters[i - 1]
            if child_kernel_sizes[i] > child_kernel_sizes[i - 1]:
                child_kernel_sizes[i] = child_kernel_sizes[i - 1]
            if child_pool_sizes[i] > child_pool_sizes[i - 1]:
                child_pool_sizes[i] = child_pool_sizes[i - 1]

        child_fcn_units = []
        for i in range(child_fcn_layers):
            unit = choose_param(parent1.fcn_units, parent2.fcn_units, i)
            if unit is None:
                unit = random.randint(32, 256)
            child_fcn_units.append(unit)

        child = DynamicNN(num_cnns=child_cnn_layers,
                          num_fcns=child_fcn_layers,
                          filters=child_filters,
                          kernel_sizes=child_kernel_sizes,
                          strides=child_strides,
                          pool_sizes=child_pool_sizes,
                          fcn_units=child_fcn_units,
                          activation_index=child_activation_index,
                          lr=child_lr).to(self.device)
        return child


    def mutate(self, individual):
        mutated = False

        lr_choices = self.lr_choices
        possible_filters = [16, 32, 64, 128]
        allowed_kernel_sizes = [3, 5, 7]
        allowed_pool_sizes = [2, 3, 4]

        if random.random() < self.mutation_rate:
            individual.num_cnns = random.randint(1, 4)
            mutated = True
        if random.random() < self.mutation_rate:
            individual.num_fcns = random.randint(1, 4)
            mutated = True
        if random.random() < self.mutation_rate:
            individual.activation_index = random.randint(0, len(self.activation_functions) - 1)
            mutated = True
        if random.random() < self.mutation_rate:
            individual.lr = random.choice(lr_choices)
            mutated = True

        current_cnn_layers = individual.num_cnns
        while len(individual.filters) < current_cnn_layers:
            if len(individual.filters) == 0:
                individual.filters.append(random.choice(possible_filters))
            else:
                allowed = [f for f in possible_filters if f <= individual.filters[-1]]
                individual.filters.append(random.choice(allowed))
        while len(individual.kernel_sizes) < current_cnn_layers:
            if len(individual.kernel_sizes) == 0:
                individual.kernel_sizes.append(random.choice(allowed_kernel_sizes))
            else:
                allowed = [k for k in allowed_kernel_sizes if k <= individual.kernel_sizes[-1]]
                individual.kernel_sizes.append(random.choice(allowed))
        while len(individual.strides) < current_cnn_layers:
            individual.strides.append(random.randint(1, 2))
        while len(individual.pool_sizes) < current_cnn_layers:
            if len(individual.pool_sizes) == 0:
                individual.pool_sizes.append(random.choice(allowed_pool_sizes))
            else:
                allowed = [p for p in allowed_pool_sizes if p <= individual.pool_sizes[-1]]
                individual.pool_sizes.append(random.choice(allowed))

        for i in range(current_cnn_layers):
            if random.random() < self.mutation_rate:
                if i == 0:
                    individual.filters[i] = random.choice(possible_filters)
                else:
                    allowed = [f for f in possible_filters if f <= individual.filters[i - 1]]
                    individual.filters[i] = random.choice(allowed)
                mutated = True
            if random.random() < self.mutation_rate:
                if i == 0:
                    individual.kernel_sizes[i] = random.choice(allowed_kernel_sizes)
                else:
                    allowed = [k for k in allowed_kernel_sizes if k <= individual.kernel_sizes[i - 1]]
                    individual.kernel_sizes[i] = random.choice(allowed)
                mutated = True
            if random.random() < self.mutation_rate:
                individual.strides[i] = random.randint(1, 2)
                mutated = True
            if random.random() < self.mutation_rate:
                if i == 0:
                    individual.pool_sizes[i] = random.choice(allowed_pool_sizes)
                else:
                    allowed = [p for p in allowed_pool_sizes if p <= individual.pool_sizes[i - 1]]
                    individual.pool_sizes[i] = random.choice(allowed)
                mutated = True

        current_fcn_layers = individual.num_fcns
        while len(individual.fcn_units) < current_fcn_layers:
            individual.fcn_units.append(random.randint(32, 256))
        for i in range(current_fcn_layers):
            if random.random() < self.mutation_rate:
                individual.fcn_units[i] = random.randint(32, 256)
                mutated = True

        if mutated:
            new_individual = DynamicNN(num_cnns=individual.num_cnns,
                                       num_fcns=individual.num_fcns,
                                       filters=individual.filters,
                                       kernel_sizes=individual.kernel_sizes,
                                       strides=individual.strides,
                                       pool_sizes=individual.pool_sizes,
                                       fcn_units=individual.fcn_units,
                                       activation_index=individual.activation_index,
                                       lr=individual.lr).to(self.device)
            individual = new_individual

        return individual

    def train_model(self, model):
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        best_val_accuracy = 0.0
        patience = 2  
        no_improvement = 0

        train_loss_list = []
        val_accuracy_list = []

        print(f"-----Training model-----")
        for epoch in range(self.epochs_per_model):
            print(f"Epoch {epoch + 1}: ...")
            model.train()
            epoch_losses = []
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(torch.log(outputs), labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            train_loss_list.append(avg_train_loss)

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            val_accuracy_list.append(val_accuracy)

            print(f"Train Loss = {avg_train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= patience:
                print("Early stopping triggered.")
                break
        
        return train_loss_list,val_accuracy_list
        
    def test_model(self, model, test_loader):
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_accuracy = test_correct / test_total if test_total > 0 else 0
        print(f"Final Test Accuracy = {test_accuracy:.4f}")

        return test_accuracy


    def calculate_fitness(self, model):
        try:
            train_loss, val_acc = self.train_model(model)
            test_acc = self.test_model(model, self.test_loader)
            
            last_train_loss = train_loss[-1] if train_loss else float('inf')
            last_val_accuracy = val_acc[-1] if val_acc else 0.0
            test_accuracy = test_acc

            fitness = (last_val_accuracy + test_accuracy) / 2.0

            print(f"Last epoch metrics - Train Loss: {last_train_loss:.4f}, Val Accuracy: {last_val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print(f"Calculated Fitness = {fitness:.4f}")

            return fitness
        except RuntimeError as e:
            print(f"RuntimeError encountered during fitness evaluation: {e}")
            return 0.0


    def save_population_metrics_history(self, population_metrics_history):
        history_dir = self.result_path / "history"
        os.makedirs(history_dir, exist_ok=True)

        filename = "population_metrics_history.json"
        file_path = history_dir /  filename # os.path.join(history_dir, "population_metrics_history.json")

        with open(file_path, 'w') as f:
            json.dump(population_metrics_history, f, indent=4)

        print(f"Population metrics history saved to: {file_path}")


    def run(self):
        if not (self.train_loader and self.val_loader and self.test_loader and self.criterion):
            raise ValueError("Data loaders and criterion must be provided in the constructor.")

        best_model_dir = self.result_path / "best_model"
        os.makedirs(best_model_dir, exist_ok=True)

        self.initialize_population(num_cnn_layers_range=(1, 4), num_fcn_layers_range=(1, 4))
        population_metrics_history = []
        previous_best_child = None
        previous_best_fitness = 0.0
        generation_without_change = 0
        best_child = None

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            generation_metrics = [] 
            fitness_scores = []

            for individual in self.population:
                try:
                    train_info = self.train_model(individual)
                    test_accuracy = self.test_model(individual, self.test_loader)
                    last_val_accuracy = train_info[1][-1] if train_info[1] else 0.0
                    fitness = (last_val_accuracy + test_accuracy) / 2.0
                except RuntimeError as e:
                    print(f"RuntimeError encountered: {e}")
                    train_info = {'train_losses': [], 'val_accuracies': [], 'test_accuracy': 0.0}
                    fitness = 0.0

                generation_metrics.append({
                    'train_losses': train_info[0],
                    'val_accuracies': train_info[1],
                    'test_accuracy': test_accuracy,
                    'fitness': fitness
                })
                fitness_scores.append(fitness)

            population_metrics_history.append(generation_metrics)

            sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
            best_index = sorted_indices[0]
            best_child = self.population[best_index]
            best_fitness = fitness_scores[best_index]
            print(f"Best fitness in generation {generation + 1}: {best_fitness:.4f}")

            filename = os.path.join(best_model_dir, f"generation_{generation + 1:02d}.pth")
            torch.save(best_child.state_dict(), filename)
            print(f"Saved best model for generation {generation + 1} as {filename}")

            num_selected = self.population_size // 2
            selected_individuals = [self.population[i] for i in sorted_indices[:num_selected]]

            new_population = selected_individuals.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

            if previous_best_child is not None:
                improvement = best_fitness - previous_best_fitness
                if improvement < 0.005 * previous_best_fitness:
                    generation_without_change += 1
                else:
                    generation_without_change = 0

            if best_fitness >= 0.9999:
                print("Achieved nearly perfect fitness. Early stopping (potential overfitting).")
                torch.save(best_child.state_dict(), os.path.join(best_model_dir, f"generation_{generation + 1:02d}_overfitting.pth"))
                if previous_best_child:
                    torch.save(previous_best_child.state_dict(), os.path.join(best_model_dir, f"generation_{generation + 1:02d}_overfitting_prev.pth"))
                break

            if generation_without_change >= 10:
                print("No significant improvement in recent generations. Early stopping.")
                torch.save(best_child.state_dict(), os.path.join(best_model_dir, f"generation_{generation + 1:02d}_no_improvement.pth"))
                break

            previous_best_child = best_child
            previous_best_fitness = best_fitness

        else:
            print("Finished all generations.")
            torch.save(best_child.state_dict(), os.path.join(best_model_dir, f"generation_{self.generations:02d}_final.pth"))

        self.save_population_metrics_history(population_metrics_history)

        return population_metrics_history
