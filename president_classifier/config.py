from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    # Parâmetros de modelo
    image_size: int = 224
    batch_size: int = 32
    num_classes: int = 6  # 5 preidentes + 1 discriminador
    freeze_layers: bool = True
    
    # Parâmetros de treinamento
    epochs: int = 5
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Parâmetros de dados
    data_dir: str = "data/presidents_cropped"