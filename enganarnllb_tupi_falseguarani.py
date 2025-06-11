import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

torch.cuda.empty_cache()

# Carrega o Excel com pares de tradução
df = pd.read_excel("tupi_antigo_limpo3.xlsx")
df = df.dropna(subset=["portugues", "Tupi antigo"])

# Converte para string
for col in ["portugues", "Tupi antigo"]:
    df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else "")

# Renomeia colunas (PT → Tupi Antigo)
df = df.rename(columns={"portugues": "translation_source", "Tupi antigo": "translation_target"})

# Converte para Dataset do HuggingFace
dataset = Dataset.from_pandas(df)

# Modelo NLLB
model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Configura idiomas corretamente (PT → fingindo Guarani para Tupi Antigo)
tokenizer.src_lang = "por_Latn"
tokenizer.tgt_lang = "grn_Latn"

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["translation_source"],
        max_length=128,
        truncation=True
    )
    labels = tokenizer(
        text_target=examples["translation_target"],
        max_length=128,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_pt_tupi_engana",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    save_total_limit=2,
    logging_dir="./logs_pt_tupi_engana",
    logging_steps=100,
    save_steps=500,
    do_train=True,
    do_eval=False,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./modelo_final_pt_tupi2")
