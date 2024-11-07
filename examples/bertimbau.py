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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier (model, tokenizer, model.bert.embeddings.word_embeddings).to(device)

"""
    print("New Attacker")
    attacker = OpenAttack.attackers.TextFoolerAttacker()

    # create your dataset here
    dataset = datasets.Dataset.from_dict({
        "x": [
            "Eu odeio este filme.",
            "Eu gosto deste filme."
        ],
        "y": [
            0, # 0 for negative
            1, # 1 for positive
        ]
    })

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)
"""    

if __name__ == "__main__":
    main()