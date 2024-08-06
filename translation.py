import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Carregue o modelo e o tokenizer
model_name = "facebook/nllb-200-distilled-1.3B"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate(text, src_lang="por", tgt_lang="tpi"):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# Carregue o arquivo Excel
df = pd.read_excel('data/Citações em tupi (1).xlsx')


df['Tupi'] = df['Tradução'].apply(lambda x: translate(x, src_lang="por", tgt_lang="tpi"))

# Salve o arquivo traduzido
df.to_excel('data/Traduzido em Tupi.xlsx', index=False)
