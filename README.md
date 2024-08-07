# Old Tupi Translator
Tupi Antigo- Tradutor

Este repositório contém um tradutor de português para tupi antigo usando o modelo `facebook/nllb-200-distilled-1.3B`.

## Como usar

1. Adicione suas frases em português no arquivo `data/Citações em tupi (1).xlsx` na coluna `Tradução`.
2. Ao fazer push no branch `main`, o GitHub Actions irá automaticamente traduzir as frases e salvar o resultado em `data/Traduzido em Tupi.xlsx`.

## Dependências

- transformers
- torch
- pandas
- openpyxl


<!-- Teste de workflow no Google Colab -->
<!-- Teste de workflow no Google Colab -->
<!-- Teste de workflow no Google Colab -->