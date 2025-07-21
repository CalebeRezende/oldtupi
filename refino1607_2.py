import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from datasets import Dataset, disable_caching

# === CONFIGURAÇÕES ===
modelo_pt2tp = "modelo_final_pt_tupi"
modelo_dados_antigos = "traduzido_pt_para_tupi_antigo_completo.xlsx"
mse_limite = 0.4
k = 5
max_len = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_caching()
torch.cuda.empty_cache()

# === FUNÇÕES ===
def traduz(textos, tokenizer, model, src_lang, tgt_lang, batch_size=50):
    tokenizer.src_lang = src_lang
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
        decodificado = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        resultados.extend(decodificado)
        torch.cuda.empty_cache()
    return resultados

def calcular_mse(lista1, lista2, embedder):
    emb1 = embedder.encode(lista1, convert_to_tensor=True, device=device)
    emb2 = embedder.encode(lista2, convert_to_tensor=True, device=device)
    return torch.mean((emb1 - emb2)**2, dim=1).cpu().numpy()

# === CARREGAR MODELOS ===
tok_pt2tp_base = AutoTokenizer.from_pretrained(modelo_pt2tp, use_fast=False)
mod_pt2tp_base = AutoModelForSeq2SeqLM.from_pretrained(modelo_pt2tp)
tok_tp2pt = AutoTokenizer.from_pretrained("modelo-nllb-tupi_port_1207_seminterrupçãoporeval", use_fast=False)
mod_tp2pt = AutoModelForSeq2SeqLM.from_pretrained("modelo-nllb-tupi_port_1207_seminterrupçãoporeval").to(device)
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(device)

# === AVALIAÇÃO DAS FRASES INÉDITAS ===
df = pd.read_csv("frases_ineditas_avaliadas.csv")
frases_pt = df["pt"].astype(str).tolist()

print("Traduzindo PT → Tupi Antigo…")
tok_pt2tp_base.src_lang = "por_Latn"
mod_pt2tp_base = mod_pt2tp_base.to(device)
tupi_preds = traduz(frases_pt, tok_pt2tp_base, mod_pt2tp_base, "por_Latn", "▁grn_Latn")
print("Traduzindo Tupi Antigo → PT (volta)…")
pt_back = traduz(tupi_preds, tok_tp2pt, mod_tp2pt, "grn_Latn", "▁por_Latn")

print("Calculando MSE…")
mse_vals = calcular_mse(frases_pt, pt_back, embedder)

df_avaliado = pd.DataFrame({
    "pt": frases_pt,
    "tupi_predito": tupi_preds,
    "pt_back": pt_back,
    "mse": mse_vals
})
df_avaliado.to_csv("avaliacao_mse_frases_ineditas.csv", index=False)

# === FILTRAR PARA REFINAMENTO ===
df_refino = df_avaliado[df_avaliado["mse"] < mse_limite].reset_index(drop=True)

# === INCLUIR DADOS ANTIGOS PARA MANTER CONHECIMENTO ===
if os.path.exists(modelo_dados_antigos):
    df_antigos = pd.read_excel(modelo_dados_antigos)
    df_antigos = df_antigos.rename(columns={"Tradução": "pt", "Tupi": "tupi_predito"})
    df_combinado = pd.concat([df_refino, df_antigos], ignore_index=True)
else:
    df_combinado = df_refino.copy()

df_combinado.to_csv("refino_com_dados_antigos.csv", index=False)

# === K-FOLD NO REFINAMENTO ===
kf = KFold(n_splits=k, shuffle=True, random_state=42)
resultados = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df_combinado), start=1):
    print(f"\n▶️ Fold {fold}")

    tok_pt2tp = AutoTokenizer.from_pretrained(modelo_pt2tp, use_fast=False)
    mod_pt2tp = AutoModelForSeq2SeqLM.from_pretrained(modelo_pt2tp).to(device)

    df_train = df_combinado.iloc[train_idx]
    df_test = df_combinado.iloc[test_idx]

    dataset_train = Dataset.from_pandas(pd.DataFrame({
        "por_Latn": df_train["pt"].tolist(),
        "grn_Latn": df_train["tupi_predito"].tolist()
    }))

    dataset_train = dataset_train.map(
        lambda x: tok_pt2tp(
            x["por_Latn"], text_target=x["grn_Latn"],
            truncation=True, max_length=max_len
        ),
        batched=True
    )

    collator = DataCollatorForSeq2Seq(tok_pt2tp, model=mod_pt2tp)
    args = Seq2SeqTrainingArguments(
        output_dir=f"modelo_refino_ineditas_k{fold}",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    trainer = Seq2SeqTrainer(
        model=mod_pt2tp,
        args=args,
        train_dataset=dataset_train,
        tokenizer=tok_pt2tp,
        data_collator=collator
    )
    trainer.train()

    frases_teste = df_test["pt"].tolist()
    tupi_pred = traduz(frases_teste, tok_pt2tp, mod_pt2tp, "por_Latn", "▁grn_Latn")
    pt_recuperado = traduz(tupi_pred, tok_tp2pt, mod_tp2pt, "grn_Latn", "▁por_Latn")
    mse_test = calcular_mse(frases_teste, pt_recuperado, embedder)

    df_avaliacao = pd.DataFrame({
        "pt": frases_teste,
        "tupi_predito": tupi_pred,
        "pt_back": pt_recuperado,
        "mse": mse_test
    })
    df_avaliacao.to_csv(f"avaliacao_k{fold}_ineditas.csv", index=False)

    resultados.append({
        "fold": fold,
        "mse_medio": round(mse_test.mean(), 4),
        "mse_min": round(mse_test.min(), 4),
        "mse_max": round(mse_test.max(), 4),
        "n_testes": len(mse_test)
    })

    mod_pt2tp.save_pretrained(f"modelo_refino_ineditas_k{fold}")
    tok_pt2tp.save_pretrained(f"modelo_refino_ineditas_k{fold}")

    del trainer, mod_pt2tp, tok_pt2tp
    torch.cuda.empty_cache()

resumo_df = pd.DataFrame(resultados)
resumo_df.to_csv("resumo_kfold_refino_ineditas.csv", index=False)
melhor_fold = resumo_df.loc[resumo_df["mse_medio"].idxmin()]
print(f"\n🏆 Melhor modelo refinado: Fold {int(melhor_fold['fold'])} com MSE médio = {melhor_fold['mse_medio']}")

print("\n✅ K-Fold com frases inéditas e dados antigos finalizado!")

