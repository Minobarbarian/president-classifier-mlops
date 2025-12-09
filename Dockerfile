# Use umma imagem oficial leve do Python
FROM python:3.10-slim

# Configure o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instale dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie o arquivo de requisitos primeiro
COPY requirements.txt .

# Instale dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação
COPY . .

# Configure o PYTHONPATH para que o Python saiba onde procurar seus módulos
ENV PYTHONPATH=/app

# Comando padrão para manter o contêiner em execução
CMD ["tail", "-f", "/dev/null"]