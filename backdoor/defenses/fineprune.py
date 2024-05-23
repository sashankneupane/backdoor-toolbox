import torch
import torch.nn as nn

class FinePrune:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)
    
    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def get_activation_hooks(self, layer_names):
        activations = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        for name in layer_names:
            layer = self.get_module_by_name(name)
            hook = layer.register_forward_hook(get_activation(name))
            hooks.append(hook)

        return activations, hooks

    def get_module_by_name(self, module_name):
        modules = module_name.split('.')
        module = self.model
        for mod in modules:
            module = module._modules.get(mod)
            if module is None:
                raise AttributeError(f"Module {mod} not found in {module_name}")
        return module

    def calculate_mean_activations(self, layer_names):
        self.model.eval()
        activations, hooks = self.get_activation_hooks(layer_names)
        mean_activations = {name: 0 for name in layer_names}
        num_batches = 0

        with torch.no_grad():
            for inputs, _ in self.train_loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                for name in layer_names:
                    mean_activations[name] += activations[name].mean(dim=(0, 2, 3))
                num_batches += 1

        for name in layer_names:
            mean_activations[name] /= num_batches

        for hook in hooks:
            hook.remove()

        return mean_activations

    def prune_based_on_activations(self, mean_activations, layer_names, prune_ratio):
        for name in layer_names:
            conv_layer = self.get_module_by_name(name)
            num_channels = conv_layer.weight.shape[0]
            num_channels_to_prune = int(prune_ratio * num_channels)

            # Get the indices of the channels with the smallest mean activations
            prune_indices = mean_activations[name].argsort()[:num_channels_to_prune]

            # Zero out the weights of the pruned channels
            conv_layer.weight.data[prune_indices, :, :, :] = 0
            if conv_layer.bias is not None:
                conv_layer.bias.data[prune_indices] = 0

    def prune(self, prune_ratio=0.25):
        layer_names = [name for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)]
        mean_activations = self.calculate_mean_activations(layer_names)
        self.prune_based_on_activations(mean_activations, layer_names, prune_ratio)

    def fine_prune(self, num_epochs=30, prune_ratio=0.25, min_val_acc=0.75):
        for epoch in range(num_epochs):
            self.prune(prune_ratio)
            train_loss = self.train_epoch()
            acc = self.test_model()

            print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {acc:.2f}')
            if acc < min_val_acc:
                print(f'Epoch: {epoch + 1}, Accuracy: {acc:.2f} - Stopping early due to accuracy drop below minimum threshold.')
                break
