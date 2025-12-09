import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb
import os
from .config import Config

class PresidentClassifier:
    def __init__(self, config: Config, load_path: str = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()
        
        if load_path and os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f"Loaded model from {load_path}")

    # Modelo
    def _build_model(self):

        # Carrega um Modelo Pré-treinado
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Congela Camadas Iniciais
        if self.config.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        
        # Redefine a Camada Final
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        return model.to(self.device)

    def train(self, train_loader, val_loader):
        # Otimizador e Função de Perda
        criterion = nn.CrossEntropyLoss()
        
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(self.model.fc.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = optim.SGD(self.model.fc.parameters(), lr=self.config.learning_rate, momentum=0.9)

        # Tracking de Melhor Acurácia
        best_acc = 0.0

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            
            # Laço de Treinamento
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Passo de Validação
            val_acc = self.evaluate(val_loader)
            
            # Salva Métricas para o Weights & Biases
            metrics = {
                "epoch": epoch,
                "loss": running_loss / len(train_loader),
                "val_accuracy": val_acc
            }
            wandb.log(metrics)
            print(f"Epoch {epoch}: {metrics}")

            # Salva o Modelo se for o Melhor até Agora
            if val_acc > best_acc:
                best_acc = val_acc
                self.save("best_model.pth")
                # Manda o artefato para o Weights & Biases
                wandb.save("best_model.pth")

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0

    def save(self, path):
        torch.save(self.model.state_dict(), path)