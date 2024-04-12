# possibili risorse:
# https://colab.research.google.com/github/tripathiaakash/DistilGPT2-Tutorial/blob/main/distilgpt2_fine_tuning.ipynb
# https://github.com/omidiu/GPT-2-Fine-Tuning/blob/main/main.ipynb
# https://github.com/SergioMG97/Finetuning-mT5-and-DistilGPT2-for-paraphrasing-tasks

from argparse import ArgumentParser
from datasets import Dataset
from utils import *

def configure_argparse() -> ArgumentParser:
    """
    Configure the ArgumentParser object for command line argument parsing.

    Returns:
        aparse (ArgumentParser): The configured ArgumentParser object.
    """
    # da cambiare i nomi TODO
    # TODO copia gli args da pearlm
    aparse = ArgumentParser()
    aparse.add_argument('action', type=str, help='Action to perform {tokenize, train, generate}')
    aparse.add_argument('-T', '--template-file', dest='template_path', type=str, help='Path to the filled template file')
    aparse.add_argument('-c', '--checkpoint-path', dest='ckp_path', type=str, help='Path to save/load checkpoints')
    #TODO non necessario
    aparse.add_argument('-t', '--tokenizer-file', dest='tok_path', type=str, help='Path to save/load tokenizer')
    aparse.add_argument('-p', '--path-file', dest='path_hop', type=str, help='Path to load paths')
    aparse.add_argument('-e', '--eval', type=str, help='String to pass to evaluate the model')
    return aparse

def read_paths(path):
    with open(path) as f:
        return f.read().strip()

# pathlm/models/lm/tokenize_dataset.py
# cambiare TOKENIZER_TYPE = "WordLevel" -> 'BPE'

def tokenize(text_path, tok_path):
    tokenizer = load_distilgpt2_tokenizer()
    text = read_paths(text_path)
    tokenizer = extend_distilgpt2_tokenizer(tokenizer, text)
    print(f'Saving tokenizer to {tok_path}')
    tokenizer.save_pretrained(tok_path)
    
def tokenize_dataset():
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    num_proc=args.nproc,
                                    remove_columns=["path"]
                                    )
    tokenized_dataset = DatasetDict({
        "train": tokenized_dataset,
    })
    # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
    check_dir(TOKENIZED_DATASET_PATH)
    tokenized_dataset.save_to_disk(
        TOKENIZED_DATASET_PATH)

def train_model():
    model = load_distilgpt2_model()
    data_collator, text_dataset = create_text_dataset()
    

if __name__ == '__main__':
    args = configure_argparse().parse_args()
    action = args.action
    if action == 'tokenize':
        print('Starting tokenizing action')
        tokenize(args.path_hop, args.tok_path)
    elif action == 'train':
        print('Training model')
        train_model()

    elif action == 'generate':
        print('not implemented yet')
    else:
        print('Invalid action. Please choose one of {tokenize, train, generate}')
        exit(1)
