import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Carrega o modelo NLLB
model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define as línguas
tokenizer.src_lang = "por_Latn"
target_lang = "grn_Latn"
forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

# Carrega o Excel
arquivo = "tupi_antigo_limpo3.xlsx"
df = pd.read_excel(arquivo)

# Define os nomes corretos das colunas
coluna_portugues = "portugues"
coluna_guarani = "Guarani"

# Garante que a coluna 'Guarani' existe
if coluna_guarani not in df.columns:
    df[coluna_guarani] = ""

# Tradução linha a linha
for idx, texto_pt in enumerate(df[coluna_portugues]):
    if pd.isna(texto_pt) or pd.notna(df.at[idx, coluna_guarani]):
        continue  # Pula se estiver vazio ou já traduzido

    print(f"🔄 Traduzindo linha {idx+1}...")

    inputs = tokenizer(str(texto_pt), return_tensors="pt")
    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id
    )
    traducao = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    df.at[idx, coluna_guarani] = traducao

# Salva novo arquivo
df.to_excel("traduzido_pt_para_guarani_limpo.xlsx", index=False)
print("✅ Traduções salvas em 'traduzido_pt_para_guaran-limpo.xlsx'")

