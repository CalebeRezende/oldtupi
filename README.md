Modelo T5 para Tradução - Guia de Fine-Tuning e Uso
Este repositório fornece um guia para o fine-tuning e uso do modelo T5 para tarefas de tradução, especificamente para traduzir do português para Tupi Antigo. O modelo é treinado usando a biblioteca transformers do Hugging Face e utiliza o Google Colab e Google Drive para salvar e carregar o modelo ajustado.

Índice
Introdução
Fine-Tuning do Modelo T5
Salvando o Modelo Treinado
Carregando o Modelo Treinado
Usando o Modelo para Tradução
Recursos Adicionais
Introdução
Para usar este projeto, você precisará do Google Colab e de uma conta no Google Drive para armazenar o modelo. Siga os passos abaixo para começar:

Clone o repositório e faça upload dos arquivos para o seu Google Drive ou ambiente do Colab.
Instale as bibliotecas necessárias com o seguinte comando no Colab ou no seu ambiente Python:
bash
Copiar código
!pip install transformers torch
Fine-Tuning do Modelo T5
O modelo T5 pode ser ajustado (fine-tuned) em conjuntos de dados personalizados. Neste exemplo, o modelo é ajustado para traduzir do português para Tupi Antigo.

Etapas para Fine-Tuning:
Monte o Google Drive no ambiente Colab para carregar e salvar o modelo e o conjunto de dados:

python

from google.colab import drive
drive.mount('/content/drive')
Carregue o conjunto de dados e prepare o pipeline de treinamento:

python

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler

# Carregar conjunto de dados do Google Drive
with open('/content/drive/MyDrive/Tupi Antigo/traduzido_tupi_antigo.json', 'r') as f:
    formatted_data = json.load(f)
Defina a classe do Dataset para preparar os dados para o modelo:

python

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
        target_text = item['expected_text']
        inputs = self.tokenizer(input_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'labels': targets['input_ids'].squeeze()}
Treine o modelo usando acumulação de gradientes e precisão mista para um treinamento eficiente:

python

def train(model, dataloader, optimizer, epochs=3, accumulation_steps=4):
    model.train()
    scaler = GradScaler()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Treinamento com precisão mista
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            print(f"Loss: {loss.item()}")
Salvando o Modelo Treinado
Após o fine-tuning do modelo, salve o modelo e o tokenizador no seu Google Drive:

python

save_directory = '/content/drive/MyDrive/Tupi Antigo/t5_fine_tuned_model/'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
Isso armazenará o modelo no diretório especificado no Google Drive.

Carregando o Modelo Treinado
Para carregar o modelo e o tokenizador posteriormente, use o código abaixo:

python

from transformers import T5Tokenizer, T5ForConditionalGeneration

save_directory = '/content/drive/MyDrive/Tupi Antigo/t5_fine_tuned_model/'
model = T5ForConditionalGeneration.from_pretrained(save_directory)
tokenizer = T5Tokenizer.from_pretrained(save_directory)
Usando o Modelo para Tradução
Depois de carregar o modelo, você pode usá-lo para traduzir novos textos de entrada. Aqui está uma função de exemplo para realizar a tradução:

python

def translate_text(input_text, model, tokenizer):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Exemplo de uso:
input_text = "Digite sua frase em português aqui."
translated_text = translate_text(input_text, model, tokenizer)
print(f"Tradução para Tupi Antigo: {translated_text}")
Recursos Adicionais
Documentação do Hugging Face Transformers
Visão Geral do Modelo T5
Documentação do Google Colab
Notas:
Ajuste os caminhos dos datasets e dos diretórios conforme necessário, dependendo da configuração do seu Google Drive.
Para modelos ou datasets maiores, pode ser necessário utilizar uma conta Colab Pro para mais memória ou disponibilidade de GPU.
