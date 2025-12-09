import os
import sys

# Adiciona a raiz do projeto ao path para poder importar 'president_classifier'
sys.path.append(os.getcwd())

from president_classifier.data import DataProcessor

# Define os caminhos dos dados
RAW = "data/presidents_raw"
CROPPED = "data/presidents_cropped"

# Define os termos de busca para cada presidente
TERMS = {
    "Lula": "Luís Inácio Lula da Silva rosto oficial",
    "Bolsonaro": "Jair Bolsonaro foto oficial rosto",
    "Temer": "Michel Temer",
    "Dilma": "Dilma Rousseff foto oficial rosto",
    "Fernando": "Fernando Henrique Cardoso rosto",
    "Alguém": "CEO headshot" 
}

if __name__ == "__main__":
    processor = DataProcessor(RAW, CROPPED, TERMS)
    processor.download_images()
    processor.crop_faces()