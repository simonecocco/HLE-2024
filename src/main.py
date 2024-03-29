# possibili risorse:
# https://colab.research.google.com/github/tripathiaakash/DistilGPT2-Tutorial/blob/main/distilgpt2_fine_tuning.ipynb
# https://github.com/omidiu/GPT-2-Fine-Tuning/blob/main/main.ipynb
# https://github.com/SergioMG97/Finetuning-mT5-and-DistilGPT2-for-paraphrasing-tasks

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForPadding, TrainingArguments, Trainer
from argparse import ArgumentParser
from datasets import Dataset

def configure_argparse():
    aparse = ArgumentParser()
    aparse.add_argument('-f', '--file', type=str, required=True, help='Path to the file')
    return aparse

def read_filled_template(path):
    with open(path) as f:
        lines = [tuple(line.strip().split('\t')) for line in f.readlines().split('\n')]
    ds = Dataset.from_dict({
        'input': [line[0].strip() for line in lines],
        'output': [line[1].strip() for line in lines]
    })


class DistilGPT2:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def encode_text(self, text, padding='max_length'):
        return self.tokenizer(text, return_tensors='pt', padding=padding)

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens)

    def preprocess_dataset(self, dataset):
        return dataset.map(self.encode_text, batched=True)

    def run(self, text):
        tokens = self.encode_text(text)
        output = self.model.generate(**tokens)
        return self.decode_tokens(output)

    def train(self):
        training_args = TrainingArguments('test-trainer')
        trainer = Trainer( # TODO
            self.model,
            training_args,

        )

if __name__ == '__main__':
    aparse = configure_argparse()
    filled_template_path = aparse.parse_args().file
    lines = read_filled_template(filled_template_path)
    gpt2 = DistilGPT2()

