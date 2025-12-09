import os
from pymongo import MongoClient
import wandb
import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "mlops_db")
COLLECTION_NAME = "model_registry"

def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

# Registra o Melhor Modelo de uma Varredura no MongoDB
def register_best_run(sweep_id, project_name):
    print(f"Encontrando melhor modelo pela varredura: {sweep_id}...")
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    # Pega a melhor execução da varredura
    best_run = sweep.best_run()
    
    print(f"🏆 Melhor Execução: {best_run.name} ({best_run.summary.get('val_accuracy'):.4f})")
    
    # Cria o documento do modelo
    model_doc = {
        "project": project_name,
        "wandb_run_id": best_run.id,
        "best_accuracy": best_run.summary.get('val_accuracy'),
        "config": best_run.config,
        "artifact_path": f"{best_run.entity}/{best_run.project}/{best_run.id}/files/best_model.pth",
        "created_at": datetime.datetime.utcnow(),
        "status": "production"
    }
    
    collection = get_collection()

    # Salva no MongoDB
    collection.update_one(
        {"project": project_name},
        {"$set": model_doc},
        upsert=True
    )
    print("✅ Registrado no MongoDB.")