'''
This example code shows how to  how to use the PWWS attack model to attack BERT on the SST-2 dataset.
'''
import ssl
import OpenAttack
import datasets

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import transformers
import torch.nn.functional as F

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
    
def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    print("Load model")
    model_name = 'lipaoMai/bert-sentiment-model-portuguese'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, output_hidden_states=True)
    #sample_text = 'que coisa linda O programa estava mostrando uma familia que adotou um adolescente de NUMBER anos que amor !'
    sample_text = 'o filme é muito ruim'
    #sample_text = 'o filme é muito bom'
    #inputs = tokenizer(sample_text)
    inputs = tokenizer.encode_plus(sample_text, max_length=512, truncation=True, padding='max_length',
                               add_special_tokens=True, return_tensors='pt')
    print(inputs.keys())
    print(inputs['input_ids'])
    output = model (**inputs)
    #print(output)
    # apply softmax to the logits output tensor of our model (in index 0) across dimension -1
    probs = F.softmax(output[0], dim=-1)
    print(probs)
    pred = torch.argmax(probs)
    print(pred.item())

""" print(f"Input tensor shape: {inputs['input_ids'].size()}")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids)
    print(tokens)
    print(f"{tokenizer.vocab_size}, {tokenizer.model_max_length}, {tokenizer.model_input_names}")
"""
   
    
"""    inputs = {k:v.to(device) for k,v in inputs.input_ids.len()}
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)
"""    

"""
with torch.no_grad():
    pred = []
    labels = []
    outputs = model(**inputs)
    pred.append(F.softmax(outputs, dim=1)[:, 1].cpu())
    labels.append(inputs['labels'].cpu())
pred = torch.cat(pred).numpy()
labels = torch.cat(labels).numpy()

print('Acc:', metrics.accuracy_score(pred>0.67, labels))
    #print(outputs.last_hidden_state.size())
"""    
if __name__ == "__main__":
    main()