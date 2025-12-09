import os
import shutil
import torch
from icrawler.builtin import BingImageCrawler
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class DataProcessor:
    def __init__(self, raw_dir, processed_dir, search_terms):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.search_terms = search_terms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def download_images(self, max_num=60):
        print(f">>> Baixando para {self.raw_dir}...")
        for label, term in self.search_terms.items():
            folder = os.path.join(self.raw_dir, label)
            os.makedirs(folder, exist_ok=True)
            
            # Baixa até ter max_num imagens
            if len(os.listdir(folder)) >= max_num:
                print(f"Skipping {label}, folder already has {len(os.listdir(folder))} images.")
                continue

            crawler = BingImageCrawler(storage={'root_dir': folder})
            crawler.crawl(keyword=term, max_num=max_num)

    def crop_faces(self):
        print(f">>> Recortando faces para {self.processed_dir}...")
        mtcnn = MTCNN(keep_all=False, select_largest=True, margin=20, device=self.device)
        
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)

        classes = [d for d in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, d))]

        for class_name in classes:
            source_dir = os.path.join(self.raw_dir, class_name)
            dest_dir = os.path.join(self.processed_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for img_name in os.listdir(source_dir):
                try:
                    img = Image.open(os.path.join(source_dir, img_name)).convert('RGB')
                    mtcnn(img, save_path=os.path.join(dest_dir, img_name))
                except Exception:
                    pass

def get_dataloaders(data_dir, batch_size, image_size):
    """
    Returna o loader de treino, validação e os nomes das classess
    """
    # Data Augmentation e Normalização
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        # Transformações geométricas (Posição/Formato)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),

        # Transformações de cor e luz
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

        # Simulação de problemas de qualidade na imagem
        transforms.RandomGrayscale(p=0.1),

        # Mudança de perspectiva
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        
        transforms.ToTensor(),

        # Apagamento aleatório
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),

        # Normalização
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Vê se a base de dados existe
    if not os.path.exists(data_dir):
         data_dir = os.path.join(os.getcwd(), data_dir)

    # Carrega o Dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Divide em Treino, Validação e Teste
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, dataset.classes