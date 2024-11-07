'''
This example code shows how to  how to use the PWWS attack model to attack BERT on the SST-2 dataset.
'''
import ssl
import OpenAttack
import datasets

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import transformers


def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
    
def main():
    ssl._create_default_https_context = ssl._create_unverified_context


    # Carregando o modelo e o tokenizador do BERTimbau
    #model_name = 'neuralmind/bert-base-portuguese-cased'
    #tokenizer = BertTokenizer.from_pretrained(model_name)
    #model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # num_labels = 2 para classificação binária (positivo/negativo)

    #print(model._get_name)
    #victim = OpenAttack.loadVictim(model._get_name)


    print("Load model")
    model_name = 'lipaoMai/bert-sentiment-model-portuguese'
    sample_text = 'que coisa linda O programa estava mostrando uma familia que adotou um adolescente de NUMBER anos que amor !'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.zeros(1).cuda()
    inputs = tokenizer(sample_text, return_tensors="pt")
    print(f"Input tensor shape: {inputs['input_ids'].size()}")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, output_hidden_states=False)
    #model = transformers.AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)
    #print(outputs.last_hidden_state.size())
    victim = OpenAttack.classifiers.TransformersClassifier (model, tokenizer, model.bert.embeddings.word_embeddings).to(device)

    print("New Attacker")
    attacker = OpenAttack.attackers.TextFoolerAttacker()



    # BERT.SST is a pytorch model which is fine-tuned on SST-2. It uses Glove vectors for word representation.
    # The load operation returns a PytorchClassifier that can be further used for Attacker and AttackEval.

#    dataset = datasets.load_dataset("sst", split="train[:20]", trust_remote_code=True).map(function=dataset_mapping)
    # We load the sst-2 dataset using `datasets` package, and map the fields.

#    attacker = OpenAttack.attackers.PWWSAttacker()
    # After this step, we’ve initialized a PWWSAttacker and uses the default configuration during attack process.

#    attack_eval = OpenAttack.AttackEval(attacker, victim)
    # Use the default implementation for AttackEval which supports seven basic metrics.

 #   attack_eval.eval(dataset, visualize=True)
    # Using visualize=True in attack_eval.eval can make it displays a visualized result. This function is really useful for analyzing small datasets.

if __name__ == "__main__":
    main()