from argparse import ArgumentParser
from datasets import Dataset
from utils import *

from torch.utils.data import DataLoader
from transformers import AutoTokenizer,\
                        AutoModelForCausalLM,\
                        DataCollatorForLanguageModeling,\
                        TextDataset,\
                        TrainingArguments,\
                        PreTrainedTokenizerFast
from datasets import DatasetDict
from tokenizers import AddedToken
import pandas as pd
from os.path import exists, join
from os import mkdir

TOK_TYPE = 'BPE'
MODEL = 'distilgpt2'

SPECIAL_TOKENS = {
        'start_pi_token':'<start_pi>',
        'end_pi_token':'<end_pi>',
        'start_rp_token':'<start_rp>',
        'end_rp_token':'<end_rp>',
        'start_se_token':'<start_se>',
        'end_se_token':'<end_se>',
        'start_te_token':'<start_te>',
        'end_te_token':'<end_te>',
        'start_re_token':'<start_re>',
        'end_re_token':'<end_re>',
        'start_rec_token':'<start_rec>',
        'end_rec_token':'<end_rec>',
        'start_exp_token':'<start_exp>',
        'end_exp_token':'<end_exp>'
    }

def load_tokenizer_from_local(tok_path):
    return AutoTokenizer.from_pretrained(tok_path)

def load_distilgpt2_model():
    return AutoModelForCausalLM.from_pretrained('distilgpt2')

def create_text_dataset(tokenizer, text):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    text_dataset = TextDataset(tokenizer=tokenizer, file_path=template, block_size=512)
    return data_collator, train, test, val

def create_training_args():
    return TrainingArguments(
        output_dir='distilgpt2_tuned',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10,
        learning_rate=1e-5
    )
    
def extend_tokenizer(tokenizer, paths):
    print(f'Adding {len(SPECIAL_TOKENS)} special tokens')
    special_tokens = [AddedToken(v, single_word=True, lstrip=True, rstrip=True, normalized=False) for v in SPECIAL_TOKENS.values()]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    kg_tokens = set([t for t in paths.split()])
    print(f'Adding {len(kg_tokens)} normal tokens')
    tokenizer.add_tokens([AddedToken(t, single_word=True, lstrip=True, rstrip=True, normalized=False) for t in kg_tokens])
    # TODO tokenizer.resize_token_embeddings(len(specials.k) + len(normal_words))
    print('Done')

    return tokenizer

def read_paths(path):
    with open(path) as f:
        return f.read().strip()

def expand_and_save_tokenizer(model_name, text_path, tok_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    paths = read_paths(text_path)
    tokenizer = extend_tokenizer(tokenizer, paths)
    print(f'Saving tokenizer to {tok_path}')
    tokenizer.save_pretrained(tok_path)
    return tokenizer
    
def tokenize_function(examples: str):
    context_len = tokenizer.model_max_length
    return tokenizer(examples["path"], max_length=context_len)

def tokenize_dataset(tokenizer, dataset, dataset_output_dir):
    #tokenizer = PreTrainedTokenizerFast(tokenizer, use_fast=False, truncation=True)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    num_proc=2,
                                    remove_columns=["path"]
                                    )
    tokenized_dataset = DatasetDict({
        "train": tokenized_dataset,
    })
    tokenized_dataset.save_to_disk(join(dataset_output_dir, 'tokenized_dataset.hf'))

if __name__ == '__main__':
    aparse = ArgumentParser()
    aparse.add_argument('--filled_templates_file', default='filled_templates.txt', type=str, help='Path to the filled template file')
    aparse.add_argument('--path_file', default='paths_end-to-end_250_3.txt', type=str, help='Path to load paths')
    aparse.add_argument('--model', default='distilgpt2', type=str, help='Model name from Hugging Face')
    aparse.add_argument('--dataset_name', default='ml1m', type=str, help='Name of the dataset')
    args = aparse.parse_args()
    # Expand the model tokenizer with special tokens and kg tokens
    tokenizer_file = f'{args.model}_tokenizer'
    tokenizer_file_path = join(get_tokenizer_dataset(args.dataset_name), tokenizer_file)
    paths_file_path = join(get_raw_paths_dir(args.dataset_name), args.path_file)
    filled_templates_file_path = join(get_filled_templates_path(args.dataset_name), args.filled_templates_file)

    tokenizer = expand_and_save_tokenizer(args.model, paths_file_path, tokenizer_file_path)
    # Tokenize the dataset
    df = pd.read_csv(filled_templates_file_path, header=None, names=["path"], index_col=None, sep='\t')
    dataset = Dataset.from_pandas(df)
    print(dataset)
    tokenized_dataset_dir = get_tokenized_dataset(args.dataset_name)
    tokenize_dataset(tokenizer, dataset, tokenized_dataset_dir)