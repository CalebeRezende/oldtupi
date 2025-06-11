
import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ⚡ Verifica se CUDA está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# 🚀 Carrega os modelos com .to(device)
nllb_name = "facebook/nllb-200-distilled-1.3B"
tokenizer_nllb = AutoTokenizer.from_pretrained(nllb_name)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

modelo_pivot_path = "modelo-nllb-guarani-tupi-comdatasettraduzidovianllb"
tokenizer_pivot = AutoTokenizer.from_pretrained(modelo_pivot_path)
model_pivot = AutoModelForSeq2SeqLM.from_pretrained(modelo_pivot_path).to(device)

modelo_direto_path = "modelo_final_pt_tupi"
tokenizer_direto = AutoTokenizer.from_pretrained(modelo_direto_path)
model_direto = AutoModelForSeq2SeqLM.from_pretrained(modelo_direto_path).to(device)

# 📥 Carrega os dados
df = pd.read_excel("traduzido_pt_para_guarani.xlsx").dropna(subset=["portugues", "Tupi antigo"])
portugues = df["portugues"].tolist()
referencias = df["Tupi antigo"].tolist()

# 🔁 Funções de tradução
def traduzir_pt_para_grn(frase):
    frase = str(frase)
    if frase.strip() == "":
        return ""
    tokenizer_nllb.src_lang = "por_Latn"
    inputs = tokenizer_nllb(frase, return_tensors="pt").to(device)
    output_tokens = model_nllb.generate(
        **inputs,
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids("grn_Latn")
    )
    return tokenizer_nllb.decode(output_tokens[0], skip_special_tokens=True)

def traduzir_grn_para_tupi(frase_grn):
    frase_grn = str(frase_grn)
    if frase_grn.strip() == "":
        return ""
    inputs = tokenizer_pivot(frase_grn, return_tensors="pt").to(device)
    output_tokens = model_pivot.generate(**inputs)
    return tokenizer_pivot.decode(output_tokens[0], skip_special_tokens=True)

def traduzir_pt_para_tupi_direto(frase):
    frase = str(frase)
    if frase.strip() == "":
        return ""
    inputs = tokenizer_direto(frase, return_tensors="pt").to(device)
    output_tokens = model_direto.generate(**inputs)
    return tokenizer_direto.decode(output_tokens[0], skip_special_tokens=True)

# 🔁 Traduções com barra de progresso
traduzido_pivotado = []
traduzido_direto = []

for frase in tqdm(portugues, desc="🔁 Traduzindo frases"):
    try:
        grn = traduzir_pt_para_grn(frase)
        tupi_pivot = traduzir_grn_para_tupi(grn)
        tupi_direto = traduzir_pt_para_tupi_direto(frase)
    except Exception as e:
        print(f"Erro ao traduzir frase: {frase}\n{e}")
        tupi_pivot, tupi_direto = "", ""

    traduzido_pivotado.append(tupi_pivot)
    traduzido_direto.append(tupi_direto)

# 📏 Função de avaliação
def coletar_metricas(nome, hipoteses, referencias):
    bleu_cla = sacrebleu.corpus_bleu(hipoteses, [referencias]).score
    chrf_cla = sacrebleu.corpus_chrf(hipoteses, [referencias]).score

    bleu_sla = []
    chrf_sla = []
    for h, r in tqdm(zip(hipoteses, referencias), total=len(hipoteses), desc=f"📏 Avaliando SLA - {nome}"):
        bleu_sla.append(sacrebleu.sentence_bleu(h, [r]).score)
        chrf_sla.append(sacrebleu.sentence_chrf(h, [r]).score)

    return {
        "BLEU_CLA": bleu_cla,
        "chrF_CLA": chrf_cla,
        "BLEU_SLA": sum(bleu_sla) / len(bleu_sla),
        "chrF_SLA": sum(chrf_sla) / len(chrf_sla)
    }

# 🚀 Avalia os dois modelos
result_pivot = coletar_metricas("Tupi via Guarani", traduzido_pivotado, referencias)
result_direto = coletar_metricas("Tupi direto", traduzido_direto, referencias)

# 📄 Cria DataFrame com as métricas
df_metricas = pd.DataFrame({
    "Métrica": ["BLEU_CLA", "chrF_CLA", "BLEU_SLA", "chrF_SLA"],
    "Tupi via Guarani": [result_pivot[m] for m in ["BLEU_CLA", "chrF_CLA", "BLEU_SLA", "chrF_SLA"]],
    "Tupi direto": [result_direto[m] for m in ["BLEU_CLA", "chrF_CLA", "BLEU_SLA", "chrF_SLA"]],
})

# 💾 Salva métricas em CSV
df_metricas.to_csv("avaliacao_metricas_tradutores.csv", index=False)

# 📊 Gera gráfico
fig, ax = plt.subplots(figsize=(10, 6))
largura = 0.35
x = range(len(df_metricas["Métrica"]))

ax.bar([i - largura/2 for i in x], df_metricas["Tupi via Guarani"], width=largura, label="Via Guarani")
ax.bar([i + largura/2 for i in x], df_metricas["Tupi direto"], width=largura, label="Direto")

ax.set_xlabel("Métrica")
ax.set_ylabel("Pontuação")
ax.set_title("Comparação de Métricas entre Tradutores")
ax.set_xticks(x)
ax.set_xticklabels(df_metricas["Métrica"])
ax.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("grafico_metricas_tradutores.png")
plt.close()

# 💾 Salva traduções em CSV
df_resultados = pd.DataFrame({
    "portugues": portugues,
    "Tupi_Referencia": referencias,
    "Tupi_via_Guarani": traduzido_pivotado,
    "Tupi_direto": traduzido_direto
})
df_resultados.to_csv("comparacao_traducoes.csv", index=False)

