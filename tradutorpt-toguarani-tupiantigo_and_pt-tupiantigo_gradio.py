from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# 🚀 1. Modelo NLLB: pt → guarani
nllb_name = "facebook/nllb-200-distilled-1.3B"
tokenizer_nllb = AutoTokenizer.from_pretrained(nllb_name)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_name)

# 🚀 2. Modelo customizado: guarani → tupi antigo
custom_model_path = "modelo-nllb-guarani-tupi-comdatasetlimpo"
tokenizer_tupi = AutoTokenizer.from_pretrained(custom_model_path)
model_tupi = AutoModelForSeq2SeqLM.from_pretrained(custom_model_path)

# 🚀 3. Modelo direto: português → tupi antigo
modelo_direto_path = "modelo_final_pt_tupi2"
tokenizer_direto = AutoTokenizer.from_pretrained(modelo_direto_path)
model_direto = AutoModelForSeq2SeqLM.from_pretrained(modelo_direto_path)

# 🔁 Funções
def traduzir_pt_para_guarani(frase):
    tokenizer_nllb.src_lang = "por_Latn"
    inputs = tokenizer_nllb(frase, return_tensors="pt")
    output_tokens = model_nllb.generate(
        **inputs,
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids("grn_Latn")
    )
    return tokenizer_nllb.decode(output_tokens[0], skip_special_tokens=True)

def traduzir_guarani_para_tupi(frase_grn):
    inputs = tokenizer_tupi(frase_grn, return_tensors="pt")
    output_tokens = model_tupi.generate(**inputs)
    return tokenizer_tupi.decode(output_tokens[0], skip_special_tokens=True)

def traduzir_pt_para_tupi_direto(frase):
    inputs = tokenizer_direto(frase, return_tensors="pt")
    output_tokens = model_direto.generate(**inputs)
    return tokenizer_direto.decode(output_tokens[0], skip_special_tokens=True)

# 🧠 Função principal para o Gradio
def traduzir_completo(frase_pt):
    frase_grn = traduzir_pt_para_guarani(frase_pt)
    frase_tupi_pivotado = traduzir_guarani_para_tupi(frase_grn)
    frase_tupi_direto = traduzir_pt_para_tupi_direto(frase_pt)
    return frase_grn, frase_tupi_pivotado, frase_tupi_direto

# 🌐 Interface Gradio
iface = gr.Interface(
    fn=traduzir_completo,
    inputs=gr.Textbox(label="Digite uma frase em Português"),
    outputs=[
        gr.Textbox(label="Guarani (via NLLB)"),
        gr.Textbox(label="Tupi Antigo (via Guarani)"),
        gr.Textbox(label="Tupi Antigo (via caminho direto)")
    ],
    title="Tradutor Multicamadas",
    description="Este tradutor usa dois caminhos: via Guarani e via modelo direto para Tupi Antigo."
)

# 🚀 Executar com link público
iface.launch(share=True)

