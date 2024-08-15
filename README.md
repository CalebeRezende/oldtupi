Este repositório contém um tradutor de português para guarani usando o modelo `facebook/nllb-200-3.3B`.

## Como usar

o file 'testeotradutor', se rodando em colab, há espaço para o input em português, que automaticamente traduz para guarani.

## Dependências

- transformers
- torch
- pandas
- openpyxl

O modelo facebook/nllb-200-distilled-600M foi treinado usando o conjunto de dados Flores-200, que é um benchmark de tradução com foco em idiomas de baixo recurso. Este conjunto de dados é uma parte central da iniciativa No Language Left Behind (NLLB), que visa melhorar a qualidade da tradução automática para uma ampla gama de idiomas, especialmente aqueles com menos dados disponíveis.

O treinamento foi feito utilizando uma abordagem baseada em Sparsely Gated Mixture of Experts (MoE), que ajuda a modelar traduções de alta qualidade mesmo para idiomas com poucos dados disponíveis. A performance do modelo foi avaliada em mais de 40.000 direções de tradução diferentes, utilizando tanto avaliações humanas quanto métricas automáticas como BLEU e chrF++.

## Título de uma possível escrita: 
translation into extinct languages ​​using the proximal languages ​​method
tradução para línguas extintas pelo método das linguagens proximais

testar> bert multilingual with lstm in output

multilingual bert com lstm na saída


tradução para línguas extintas pelo método das linguagens proximais
