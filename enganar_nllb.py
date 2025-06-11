import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
torch.cuda.empty_cache()

from transformers import DataCollatorForSeq2Seq
# Caminho do arquivo
df = pd.read_excel("tupi_antigo_limpo3.xlsx")
df = df.dropna(subset=["Guarani", "Tupi antigo"])

# Converte colunas para string
# Converte apenas valores não-nulos e força string, convertendo booleanos corretamente
for col in ["portugues", "Guarani", "Tupi antigo"]:
    df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else "")

# Renomeia colunas para padrão HuggingFace
df = df.rename(columns={"Guarani": "translation_source", "Tupi antigo": "translation_target"})

# Converte para Dataset
dataset = Dataset.from_pandas(df)

# Modelo NLLB
model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Função de pré-processament

tokenizer.src_lang = "grn_Latn"  # fingindo que os dados de entrada são Guarani
tokenizer.tgt_lang = "grn_Latn"  # fingindo que os dados de saída também são Guarani

def preprocess_function(examples):
    # Codifica o texto de entrada (guarani fingido)
    model_inputs = tokenizer(
        examples["translation_source"],
        max_length=128,
        truncation=True
    )

    # Codifica o texto de saída (tupi antigo fingido como guarani)
    labels = tokenizer(
        text_target=examples["translation_target"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplica a tokenização
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Argumentos de treinofrom transformers
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    do_train=True,
    do_eval=False  # coloque True se for usar validação

)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Treinamento
trainer.train()

trainer.save_model("modelo-nllb-guarani-tupi-comdatasetlimpo")

