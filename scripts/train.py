import sys
import os
sys.path.append(os.getcwd())

import wandb
from president_classifier.config import Config
from president_classifier.model import PresidentClassifier
from president_classifier.data import get_dataloaders

def train_entrypoint():
    # Inicializa o Weights & Biases
    # Configuração padrão, será sobrescrita pela varredura do Weights & Biases
    run = wandb.init(project="president-classifier", config=Config().__dict__)
    
    # 2. Convert W&B Dict back to Config Object
    # We allow the sweep to override default Config values

    # Converte o dicionário de configuração do Weights & Biases de volta para o objeto Config
    cfg = Config(**wandb.config)
    
    # Prepara os DataLoaders
    train_loader, val_loader, classes = get_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.image_size
    )

    # Atualiza a configuração com o número real de classes encontradas nos dados
    cfg.num_classes = len(classes)
    
    # Inicializa e Treina o Modelo
    classifier = PresidentClassifier(cfg)
    classifier.train(train_loader, val_loader)

if __name__ == "__main__":
    train_entrypoint()