import sys
import os
import ssl

sys.path.append(os.getcwd())

import OpenAttack
from datasets import load_dataset


def dataset_mapping(data):
    return {
        "x": data["text"],
        "y": data["label"]
    }


def main():
    print("Loading Attacker...")
    ssl._create_default_https_context = ssl._create_unverified_context


    print("Loading Victim ...")
    victim = OpenAttack.loadVictim("BERTimbau+TTSBR")

    print("Loading Dataset ...")
    path_var=os.path.join(os.getcwd(), "data", "Dataset.Loader", "TTSBR.py")
    print(path_var)
    dataset = load_dataset(path=path_var, split="validation",
                           trust_remote_code=True).map(function=dataset_mapping)

    print("Start Attack!")
    attacker = OpenAttack.attackers.TextFoolerAttacker()
    attack_eval = OpenAttack.AttackEval(attacker, victim)
    attack_eval.eval(dataset, visualize=True, progress_bar=True)


if __name__ == "__main__":
    main()
