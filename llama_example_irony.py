from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import transformers
import torch
import os
import pandas as pd
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

access_token = "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"

model = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=access_token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=access_token
)

prompt_template = f"Geef voor elke Tweet aan of de tekst ironie (of sarcasme) bevat of niet.\
     Gebruik hiervoor als labels '1 ironie/sarcasme' als de tekst ironie of sarcasme bevat of '0 niet ironisch' als de tekst geen ironie of sarcasme bevat.\
    \n Volg voor deze taak het volgende patroon.\
    \n Tweet: Dit is een ironische tweet ;)\
    \n Antwoord: 1 ironisch/sarcastisch\
    \n Doe dit nu zelf voor de volgende tweet\
    \n Tweet: "

ironydata = pd.read_csv("ironydata/irony_traindata.csv")
ironysamples = ironydata["text"].tolist()

generative_predictions = []
for textsample in ironysamples:
    prompt = prompt_template + textsample + "\nAntwoord: "
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=5000,
    )
    for seq in sequences:
        # print(f"Result: {seq['generated_text']}")
        classificationoutput = seq['generated_text'].split(textsample + "\nAntwoord: ", )[1]
        print("classification: ", classificationoutput)
        generative_predictions.append(classificationoutput[0])

ironydata["generated_predictions"] = generative_predictions
ironydata.to_csv("ironydata/trainset_output.csv", index=False)
print("Script Complete")
