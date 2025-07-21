from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#o completo
# 🚀 1. Modelo NLLB: pt → guarani
nllb_name = "facebook/nllb-200-distilled-1.3B"
tokenizer_nllb = AutoTokenizer.from_pretrained(nllb_name)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_name)

# 🚀 2. Modelo customizado: guarani → tupi antigo
custom_model_path = "modelo-nllb-guarani-tupi-comdatasettraduzidovianllb"
tokenizer_tupi = AutoTokenizer.from_pretrained(custom_model_path)
model_tupi = AutoModelForSeq2SeqLM.from_pretrained(custom_model_path)

# 🚀 3. Modelo direto: português → tupi antigo (refinado com "camuflagem")
modelo_direto_path = "./modelo_final_pt_tupi"
tokenizer_direto = AutoTokenizer.from_pretrained(modelo_direto_path)
model_direto = AutoModelForSeq2SeqLM.from_pretrained(modelo_direto_path)

# 🔁 Traduzir do português para guarani (com NLLB)
def traduzir_pt_para_guarani(frase):
    tokenizer_nllb.src_lang = "por_Latn"
    inputs = tokenizer_nllb(frase, return_tensors="pt")
    output_tokens = model_nllb.generate(
        **inputs,
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids("grn_Latn")
    )
    return tokenizer_nllb.decode(output_tokens[0], skip_special_tokens=True)

# 🔁 Traduzir do guarani para tupi antigo (modelo treinado)
def traduzir_guarani_para_tupi(frase_grn):
    inputs = tokenizer_tupi(frase_grn, return_tensors="pt")
    output_tokens = model_tupi.generate(**inputs)
    return tokenizer_tupi.decode(output_tokens[0], skip_special_tokens=True)

# 🔁 Traduzir diretamente do português para tupi (modelo refinado)
def traduzir_pt_para_tupi_direto(frase):
    inputs = tokenizer_direto(frase, return_tensors="pt")
    output_tokens = model_direto.generate(**inputs)
    return tokenizer_direto.decode(output_tokens[0], skip_special_tokens=True)

# 🧪 Executar os dois caminhos
frase_pt = input("Digite a frase em português:\n")

# Caminho 1: Pivotado (pt → grn → tupi)
frase_grn = traduzir_pt_para_guarani(frase_pt)
frase_tupi_pivotado = traduzir_guarani_para_tupi(frase_grn)

# Caminho 2: Direto (pt → tupi)
frase_tupi_direto = traduzir_pt_para_tupi_direto(frase_pt)

# 📋 Resultados
print(f"\n🟡 Guarani intermediário: {frase_grn}")
print(f"🟢 Tupi Antigo via pivotagem: {frase_tupi_pivotado}")
print(f"🔵 Tupi Antigo via modelo direto: {frase_tupi_direto}")

