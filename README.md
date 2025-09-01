🌿 Tupi Translator – Dissertação / Dissertation Repository

Este repositório reúne todos os arquivos e experimentos desenvolvidos para a dissertação de mestrado “Tupi Antigo: Desenvolvimento de Ferramentas Computacionais para Tradução e Preservação Cultural” (IME-USP, 2025).
This repository gathers all files and experiments developed for the master’s dissertation “Old Tupi: Development of Computational Tools for Translation and Cultural Preservation” (IME-USP, 2025).

O trabalho investiga métodos de treinamento, fine-tuning e avaliação de modelos de linguagem (LLaMA/NLLB) para tradução automática Português ↔ Tupi Antigo, com ênfase em estratégias de baixo recurso.
The work investigates methods of training, fine-tuning, and evaluation of language models (LLaMA/NLLB) for Portuguese ↔ Old Tupi machine translation, with an emphasis on low-resource strategies.

📂 Estrutura do Repositório / Repository Structure

datasets/
Contém todos os conjuntos de dados utilizados para treinamento e avaliação.
Contains all datasets used for training and evaluation.

portugues_tupi.json – Corpus paralelo principal / Main parallel corpus.

traduzido_pt_para_guarani_limpo.xlsx – Dataset pivotado via Guarani / Pivoted dataset via Guarani.

frases_ineditas.xlsx – Frases inéditas para avaliação final / Unseen sentences for final evaluation.

scripts/
Implementações em Python utilizadas ao longo do projeto.
Python implementations used throughout the project.

finetune_llama.py – Script de fine-tuning do modelo LLaMA / Fine-tuning script for LLaMA model.

pipeline_kfold_refino_mse.py – Treinamento com validação K-Fold + refinamento por MSE / K-Fold training with MSE-based refinement.

avaliar_traducoes.py – Cálculo de métricas BLEU, chrF, TER, BLEURT / Computation of BLEU, chrF, TER, BLEURT metrics.

tradutor_gradio.py – Interface de tradução interativa via Gradio / Interactive translation interface with Gradio.

htmls/
Interfaces experimentais desenvolvidas para visualização de resultados.
Experimental interfaces developed for results visualization.

Protótipos de tradutores em navegador / Browser-based translator prototypes.

Ferramentas de comparação entre traduções (direto vs. pivotado) / Comparison tools for direct vs. pivoted translations.

modelos/
Versões dos modelos ajustados ao longo do processo.
Versions of the models fine-tuned during the process.

modelo_final_pt_tupi/ – Modelo direto PT → Tupi após refinamento / Direct PT → Tupi model after refinement.

modelo-nllb-guarani-tupi/ – Modelo pivotado Guarani → Tupi / Pivoted Guarani → Tupi model.

modelo_tupi_port/ – Modelo para caminho inverso Tupi → Português / Reverse Tupi → Portuguese model.

⚙️ Fluxo de Trabalho / Workflow

Pré-processamento / Preprocessing

Limpeza e normalização dos datasets / Cleaning and normalization of datasets.

Tokenização personalizada para Tupi Antigo / Custom tokenization for Old Tupi.

Treinamento / Training

Fine-tuning no modelo LLaMA da Meta AI / Fine-tuning on Meta AI’s LLaMA model.

Estratégias de pivotagem via Guarani / Pivoting strategies via Guarani.

Uso de K-Fold (k=5) para avaliação robusta / Use of K-Fold (k=5) for robust evaluation.

Refinamento / Refinement

Seleção de pares com MSE < 0.4 para retreinamento / Selection of pairs with MSE < 0.4 for retraining.

Teste final em frases inéditas / Final test on unseen sentences.

Avaliação / Evaluation

Métricas: BLEU, chrF, TER, BLEURT / Metrics: BLEU, chrF, TER, BLEURT.

Comparações entre modelos direto e pivotado / Comparisons between direct and pivoted models.

Interface / Interface

Protótipo em Gradio/HTML para experimentação com usuários / Prototype in Gradio/HTML for user experimentation.

📊 Resultados / Results

O modelo PT → Tupi (refinado) apresentou melhora significativa em BLEU e chrF.

The PT → Tupi (refined) model showed significant improvement in BLEU and chrF.

O modelo Tupi → PT manteve desempenho estável, sem ganhos notáveis.

The Tupi → PT model maintained stable performance, with no significant gains.

A estratégia pivotada (PT → Guarani → Tupi) mostrou-se viável, mas menos precisa que o caminho direto.

The pivoted strategy (PT → Guarani → Tupi) proved viable but less accurate than the direct path.

🛠️ Dependências / Dependencies

Python 3.10+

Transformers

Torch

Pandas

Openpyxl

Gradio

📌 Observações / Notes

Este repositório serve como acervo integral da dissertação.

This repository serves as the integral archive of the dissertation.

Modelos finais estão disponíveis em /modelos/ e podem ser carregados no Hugging Face ou localmente.

Final models are available in /modelos/ and can be loaded in Hugging Face or locally.

O corpus está sendo documentado para futura disponibilização pública, respeitando critérios éticos e culturais.

The corpus is being documented for future public release, respecting ethical and cultural criteria.
