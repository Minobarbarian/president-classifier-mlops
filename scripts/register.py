import sys
import os
sys.path.append(os.getcwd())
from db.registry import register_best_run

if __name__ == "__main__":
    print("Cole o ID da varredura:")
    sweep_id = input().strip()
    register_best_run(sweep_id, "president-classifier")