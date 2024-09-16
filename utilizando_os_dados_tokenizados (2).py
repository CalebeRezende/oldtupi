# -*- coding: utf-8 -*-
"""utilizando os dados tokenizados

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w4i5G8jPhJ1Y5NsOFa7cc0dBEV8hLUp3
"""

!pip install transformers datasets

import json

with open('/content/drive/MyDrive/Tupi Antigo/dict-conjugated.json', 'r') as f:
    data = json.load(f)

# Transformar 'data' em um dataset que o modelo possa usar

import json

# Carregar os dados do JSON
with open('/content/drive/MyDrive/Tupi Antigo/dict-conjugated.json', 'r') as f:
    data = json.load(f)

# Transformar os dados no formato input_text -> target_text
formatted_data = []
for entry in data:
    if entry["f"] and entry["d"]:  # Verifica se há dados nas chaves 'f' e 'd'
        formatted_data.append({
            "input_text": entry["f"],    # Palavra em português
            "target_text": entry["d"]    # Tradução ou definição em Tupi Antigo
        })

# Exemplo de como ficaria a primeira entrada
print(formatted_data[0])

# Agora você pode usar 'formatted_data' no processo de tokenização e treinamento

!pip install transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carregar o tokenizer e o modelo T5 pré-treinado
model_name = "t5-small"  # Ou você pode escolher "t5-base" ou "t5-large" para um modelo maior
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

import json

# Carregar os dados do JSON
with open('/content/drive/MyDrive/Tupi Antigo/dict-conjugated.json', 'r') as f:
    data = json.load(f)

# Transformar os dados no formato input_text -> target_text
formatted_data = []
for entry in data:
    if "f" in entry and "d" in entry:  # Certifica-se de que 'f' e 'd' existem
        formatted_data.append({
            "input_text": entry["f"],    # Palavra em português
            "target_text": entry["d"]    # Tradução ou definição em Tupi Antigo
        })

# Exemplo de como ficaria a primeira entrada
print(formatted_data[0])

# Função para tokenizar os dados
def tokenize_data(data):
    input_texts = [item["input_text"] for item in data]  # Usar a chave 'input_text' que foi criada
    target_texts = [item["target_text"] for item in data]  # Usar a chave 'target_text' que foi criada

    # Tokenizar as entradas e saídas
    inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(target_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    return inputs, targets

# Tokenizar os dados carregados do JSON formatado
inputs, targets = tokenize_data(formatted_data)

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carregar o tokenizador e o modelo T5 pré-treinados
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Você pode ajustar para "t5-base", "t5-large", etc.
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Carregar os dados do JSON gerado
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    data = json.load(f)


# Exibir um exemplo para verificar o conteúdo
print(formatted_data[0])

# Função para tokenizar os dados
def tokenize_data(data):
    input_texts = [item["input_text"] for item in data]  # Usar a chave 'input_text' que foi criada
    target_texts = [item["target_text"] for item in data]  # Usar a chave 'target_text' que foi criada

    # Preparar os dados para o T5
    inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(target_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    return inputs, targets

# Tokenizar os dados carregados do JSON formatado
inputs, targets = tokenize_data(formatted_data)

# Exemplo de geração de texto com o T5 usando um dos inputs tokenizados
input_ids = inputs['input_ids'][0].unsqueeze(0)  # Selecionar o primeiro exemplo de entrada
output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

# Decodificar e exibir o texto gerado
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Texto gerado:", generated_text)

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler

# Carregar o tokenizador e o modelo T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Carregar os dados diretamente do arquivo já formatado
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    formatted_data = json.load(f)

# Verificar se o arquivo JSON foi carregado corretamente e contém dados
if not formatted_data:
    raise ValueError("O arquivo JSON está vazio ou não foi carregado corretamente.")

# Criar um Dataset para os dados
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']
        target_text = item['expected_text']  # Ajuste conforme seu JSON

        # Tokenizar as entradas e saídas
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Criar um DataLoader para o treinamento
train_dataset = TranslationDataset(formatted_data, tokenizer)

# Verificar se o dataset não está vazio
if len(train_dataset) == 0:
    raise ValueError("O dataset está vazio. Verifique os dados fornecidos.")

# Reduzir o batch_size para evitar estouro de memória
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Ajuste o batch_size conforme necessário

# Definir otimizador e parâmetros de treinamento
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Usar escalador para mixed precision
scaler = GradScaler()

# Função de treinamento com gradient accumulation e mixed precision
def train(model, dataloader, optimizer, epochs=3, accumulation_steps=4):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Usar precisão mista com autocast
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps  # Dividir o loss pelo número de acumulações

            # Backpropagation com precisão mista
            scaler.scale(loss).backward()

            # Acumulação de gradientes
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Exibir a perda (loss)
            print(f"Loss: {loss.item()}")

# Colocar o modelo na GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Treinar o modelo com gradient accumulation e mixed precision
train(model, train_loader, optimizer)

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch

# Carregar o tokenizador e o modelo T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Carregar os dados diretamente do arquivo já formatado
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    formatted_data = json.load(f)

# Verificar se o arquivo JSON foi carregado corretamente e contém dados
if not formatted_data:
    raise ValueError("O arquivo JSON está vazio ou não foi carregado corretamente.")

# Criar um Dataset para os dados
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']

        # Tokenizar as entradas e saídas
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        expected = self.tokenizer(item['expected_text'], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Criar um DataLoader para o treinamento
train_dataset = TranslationDataset(formatted_data, tokenizer)

# Verificar se o dataset não está vazio
if len(train_dataset) == 0:
    raise ValueError("O dataset está vazio. Verifique os dados fornecidos.")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Definir otimizador e parâmetros de treinamento
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Função de treinamento
def train(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            # Movendo os dados para a GPU se disponível
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Realizando a previsão
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

# Colocar o modelo na GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Treinar o modelo
train(model, train_loader, optimizer)

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch

# Carregar o tokenizador e o modelo T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Carregar os dados do JSON
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    data = json.load(f)

# Verificar se o arquivo JSON foi carregado corretamente
if not data:
    raise ValueError("O arquivo JSON está vazio ou não foi carregado corretamente.")

# Preparar os dados no formato esperado pelo modelo T5
formatted_data = []
for entry in data:
    if "f" in entry and "d" in entry:  # Certifica-se de que 'f' e 'd' existam
        formatted_data.append({
            "input_text": f"translate Portuguese to Tupi Antigo: {entry['f']}",
            "target_text": entry['d']
        })

# Verificar se o formatted_data contém entradas
if not formatted_data:
    raise ValueError("Os dados formatados estão vazios. Verifique se o JSON contém dados válidos.")

# Criar um Dataset para os dados
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']
        target_text = item['target_text']

        # Tokenizar as entradas e saídas
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Criar um DataLoader para o treinamento
train_dataset = TranslationDataset(formatted_data, tokenizer)

# Verificar se o dataset não está vazio
if len(train_dataset) == 0:
    raise ValueError("O dataset está vazio. Verifique os dados fornecidos.")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Definir otimizador e parâmetros de treinamento
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Função de treinamento
def train(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            # Movendo os dados para a GPU se disponível
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Realizando a previsão
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

# Colocar o modelo na GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Treinar o modelo
train(model, train_loader, optimizer)

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

# Carregar o tokenizador e o modelo T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Carregar os dados do JSON
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    data = json.load(f)

# Preparar os dados no formato esperado pelo modelo T5
formatted_data = []
for entry in data:
    if "f" in entry and "d" in entry:  # Certifica-se de que 'f' e 'd' existam
        formatted_data.append({
            "input_text": f"translate Portuguese to Tupi Antigo: {entry['f']}",
            "target_text": entry['d']
        })

# Criar um Dataset para os dados
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']
        target_text = item['target_text']

        # Tokenizar as entradas e saídas
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Criar um DataLoader para o treinamento
train_dataset = TranslationDataset(formatted_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Definir otimizador e parâmetros de treinamento
optimizer = AdamW(model.parameters(), lr=5e-5)

# Função de treinamento
def train(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            # Movendo os dados para a GPU se disponível
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Realizando a previsão
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

# Colocar o modelo na GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Treinar o modelo
train(model, train_loader, optimizer)

import json

# Carregar os dados do JSON
with open('/content/drive/MyDrive/Tupi Antigo/dict-conjugated.json', 'r') as f:
    data = json.load(f)

# Transformar os dados no formato input_text -> target_text
formatted_data = []
for entry in data:
    if "f" in entry and "d" in entry:  # Certifica-se de que 'f' e 'd' existem
        formatted_data.append({
            "input_text": entry["f"],    # Palavra em português
            "target_text": entry["d"]    # Tradução ou definição em Tupi Antigo
        })

# Exemplo de como ficaria a primeira entrada
print(formatted_data[0])

# Função para tokenizar os dados
def tokenize_data(data):
    input_texts = [item["input_text"] for item in data]  # Usar a chave 'input_text' que foi criada
    target_texts = [item["target_text"] for item in data]  # Usar a chave 'target_text' que foi criada

    # Tokenizar as entradas e saídas
    inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(target_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    return inputs, targets

# Tokenizar os dados carregados do JSON formatado
inputs, targets = tokenize_data(formatted_data)

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carregar o tokenizer e o modelo T5 pré-treinado
model_name = "t5-small"  # Ou você pode escolher "t5-base" ou "t5-large" para um modelo maior
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Função para formatar os dados para o T5
def format_data_for_translation(data):
    formatted_data = []
    for entry in data:
        if "f" in entry and "d" in entry:  # Certifica-se de que 'f' e 'd' existem
            formatted_data.append({
                "input_text": f"translate Portuguese to Tupi: {entry['f']}",  # Formato para T5
                "target_text": entry["d"]
            })
    return formatted_data

# Formatar os dados carregados
formatted_data = format_data_for_translation(data)

import pandas as pd
import json

# Função para traduzir texto usando o modelo T5
def translate_text(input_texts, max_length=50):
    """
    Traduz um lote de textos usando o modelo T5.

    Args:
        input_texts (list): Lista de textos para traduzir.
        max_length (int): Tamanho máximo da sequência gerada.

    Returns:
        list: Lista de textos traduzidos.
    """
    try:
        # Tokenizar o lote de textos de entrada
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids

        # Gerar as traduções
        outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)

        # Decodificar as traduções
        translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return translated_texts

    except Exception as e:
        print(f"Erro ao traduzir os textos: {str(e)}")
        return []

# Traduzir os dados formatados em lotes
batch_size = 8  # Número de exemplos por lote
translated_results = []

for i in range(0, len(formatted_data), batch_size):
    # Obter um lote de dados
    batch = formatted_data[i:i + batch_size]

    # Extrair os textos de entrada
    input_texts = [entry["input_text"] for entry in batch]

    # Traduzir o lote de textos
    translated_texts = translate_text(input_texts)

    # Armazenar os resultados
    for j, entry in enumerate(batch):
        translated_results.append({
            "input_text": entry["input_text"],
            "translated_text": translated_texts[j] if j < len(translated_texts) else "",
            "expected_text": entry["target_text"]
        })

        # Exibir as traduções (opcional)
        print(f"Entrada: {entry['input_text']}")
        print(f"Tradução gerada: {translated_texts[j] if j < len(translated_texts) else 'Erro na tradução'}")
        print(f"Tradução esperada: {entry['target_text']}")
        print()

# Salvar os resultados como CSV no Google Drive
df = pd.DataFrame(translated_results)
csv_path = '/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.csv'
df.to_csv(csv_path, index=False)
print(f"Traduções salvas com sucesso em: {csv_path}")

# Salvar os resultados como JSON no Google Drive
json_path = '/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(translated_results, f, ensure_ascii=False, indent=4)
print(f"Traduções salvas com sucesso em: {json_path}")

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Carregar o modelo T5 pré-treinado
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Dataset personalizado para o Trainer
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx]
        }

# Criar um subconjunto dos dados para treinamento mais rápido
subset_size = 1000  # Reduzir o conjunto de dados para 1000 amostras (ajustável)
train_dataset = CustomDataset(inputs[:subset_size], targets[:subset_size])

# Definir os argumentos de treinamento com otimizações
training_args = TrainingArguments(
    output_dir="./results",          # Diretório de saída
    learning_rate=2e-5,              # Taxa de aprendizado
    per_device_train_batch_size=4,   # Batch size reduzido para otimizar memória
    per_device_eval_batch_size=4,    # Batch size para avaliação também reduzido
    num_train_epochs=1,              # Apenas 1 época para reduzir tempo de treinamento
    weight_decay=0.01,               # Decaimento de peso para evitar overfitting
    save_steps=10_000,               # Passos para salvar o modelo
    save_total_limit=2,              # Limite de salvamentos para economizar espaço
    fp16=True,                       # Treinamento de precisão mista para reduzir memória
    logging_dir='./logs',            # Diretório de logs
    logging_steps=500,               # Número de passos para registrar log
    evaluation_strategy="no"         # Desativar a avaliação durante o treinamento
)

# Configurar o Trainer com o dataset e os argumentos de treinamento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=train_dataset,  # Não usar avaliação no momento
)

# Iniciar o treinamento (fine-tuning) com as otimizações
trainer.train()

# Após o treinamento, o modelo será salvo automaticamente no diretório "./results"

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Carregar o modelo T5 pré-treinado
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Dataset personalizado para o Trainer
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx]
        }

# Criar o dataset para treinamento
train_dataset = CustomDataset(inputs, targets)

# Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# Configurar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset  # Pode dividir uma parte dos dados para validação, se preferir
)

# Iniciar o treinamento (fine-tuning)
trainer.train()