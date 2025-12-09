# Classificador de Presidentes Brasileiros - Pipeline de MLOps

Esse projeto implementa uma pipeline de MLOps para classificar imagens de presidentes brasileiros. O projeto utiliza **PyTorch**, **Weights & Biases (W&B)** para otimização de hiperparâmetros, e **MongoDB** para registro de modelos.

## 🏛️ Estrutura do Projeto
Esse projeto segue o padrão "Model-View-Controller" (MVC) adaptado para Aprendizado de Máquina:

```shell
.                       
├── president_classifier/       # Lógica de Modelo
│   ├── config.py               ## Dataclass para hiperparâmetros
│   ├── data.py                 ## Crawler e Recorte Facial
│   └── model.py                ## Wrapper do PyTorch ResNet18 (Treino/Eval/Save)
├── db/                         # Camada de Persistência
│   └── registry.py             ## Conexão com MongoDB e lógica de registro
├── scripts/                    # Controladores
│   ├── prepare.py              ## Script: Baixa e processa as imagens
│   ├── train.py                ## Script: Varredura do Weights & Biases
│   └── register.py             ## Script: Registra o melhor modelo
├── docker-compose.yml          # App + MongoDB + MongoExpress
├── Dockerfile                  # Instruções para montar o ambiente
├── requirements.txt            # Dependências
├── sweep.yaml                  # Configurações das Varreduras
└── .env                        # Variáveis de Ambiente
```

## 🚀 Setup & Instalação

### 1. Variáveis de Ambiente

Crie um arquivo chamado .env no diretório raíz. Segue exemplo:
```bash
# Configurações do Weights & Biases
WANDB_API_KEY=sua_chave_aqui
WANDB_PROJECT=seu_projeto_aqui

# Configurações do Banco de Dados
MONGO_URI=mongodb://localhost:27017
MONGO_DB=seu_nome_do_banco_aqui 

# Configuração de Ambiente
ENV=dev_ou_prod
```

### 2. Infraestrutura (Docker)
Rode esse projeto no Docker.
#### Construindo e Levantando os Serviços
```bash
sudo docker-compose up -d --build
```
* App Container: president_mlops_app (Python 3.10 environment)
* Database: mlops_mongo (MongoDB)
* DB Viewer: mlops_mongo_express (Acessível em http://localhost:8081)

#### Verifique os Contêiners:
```bash
sudo docker ps
```

## 🛠️ O Workflow do MLOps
Execute os comandos abaixo dentro do contêiner do Docker.
### 1: Preparação dos Dados
Baixa imagens com Bing e recorta faces usando MTCNN.
```bash
sudo docker-compose exec app python scripts/prepare.py
```

### 2: Otimização de Hiperparâmetros(Varredura)
Ao invés de treinar apenas uma vez, rode a Varredura do Weights & Biases para encontrar a melhor configuração.

#### Inicialize a Varredura:
```bash
sudo docker-compose exec app wandb sweep sweep.yaml
```
Copie o ID da Varredura que foi retornado (e.g., username/project/xyz123).
#### Inicie o Agente:
O agente vai puxar parâmetros do Weights & Biases e executar o treinamento sem parar
```bash
sudo docker-compose exec app wandb agent <COLE_AQUI>
```
Você pode parar a execução quando quiser com `CTRL+C`.

### 3: Registro de Modelo
Quando a varredura encontrar a melhor configuração, registre-a e o caminho do artefato para o MongoDB para futuro uso.
```bash
sudo docker-compose exec app python scripts/register.py
```
Cole o ID da Varredura se pedir.
### 4: Verificação
Abra seu navegador a url http://localhost:8081 (Mongo Express). Navegue para mlops_db -> model_registry para ver seu documento de modelo.
(login: admin, senha: pass)

## 💻 Desenvolvimento Local (Opcional)
Se preferir rodar os scripts diretamente, siga os passos:
### 1: Instale Dependências
```bash
pip install -r requirements.txt
```

### 2: Inicie Somente a Base de Dados
```bash
docker-compose up -d mongo mongo-express
```

### 3: Rode os Scripts
Python vai usar localhost do seu arquivo .env.
```bash
python scripts/prepare.py
python scripts/train.py  # Esse script só roda uma execução com a configuração padrão (não faz a varredura)
python scripts/register.py
```

## 📊 Monitoramento
* Métricas de Treino: Veja em tempo real as curvas de perda e acurácia no seu Dashboard do Weights & Biases
* Base de Dados: Veja os modelos registrados em http://localhost:8081.