from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, DataCollatorForLanguageModeling, TextDataset, TrainingArguments
from tokenizers import AddedToken

TOK_TYPE = 'BPE'
MODEL = 'distilgpt2'

"""
def load_tokenizer(tokenizer_file, max_len=512):
    if '.json' != tokenizer_file[-5:]:
        raise ValueError('Tokenizer file must be a JSON file')
    if not os.path.exists(tokenizer_file) or not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f'File {tokenizer_file} not found')
    
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                   max_len=max_len,
                                   unk_token='[UNK]',
                                   pad_token='[PAD]',
                                   start_pi_token='<start_pi>',
                                   end_pi_token='<end_pi>',
                                   start_rp_token='<start_rp>',
                                   end_rp_token='<end_rp>',
                                   start_se_token='<start_se>',
                                   end_se_token='<end_se>',
                                   start_te_token='<start_te>',
                                   end_te_token='<end_te>',
                                   start_re_token='<start_re>',
                                   end_re_token='<end_re>',
                                   start_rec_token='<start_rec>',
                                   end_rec_token='<end_rec>',
                                   start_exp_token='<start_exp>',
                                   end_exp_token='<end_exp>'
                                   )
"""

def load_distilgpt2_tokenizer():
    return AutoTokenizer.from_pretrained('distilgpt2')

def load_tokenizer_from_local(tok_path):
    return AutoTokenizer.from_pretrained(tok_path)

def load_distilgpt2_model():
    return AutoModelForCausalLM.from_pretrained('distilgpt2')

def create_text_dataset(tokenizer, template):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    text_dataset = TextDataset(tokenizer=tokenizer, file_path=template, block_size=512)
    return data_collator, text_dataset

def create_training_args():
    return TrainingArguments(
        output_dir='distilgpt2_tuned',
        num_train_epochs=5
        per_device_train_batch_size=8,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10,
        learning_rate=1e-5
    )

def set
    
def extend_distilgpt2_tokenizer(tokenizer, text):
    specials = {
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
    print(f'Adding {len(specials)} special tokens')
    specials = [AddedToken(v, single_word=True, lstrip=True, rstrip=True, normalized=False) for v in specials.values()]
    tokenizer.add_special_tokens({'additional_special_tokens': specials})
    normal_words = set([t for t in text.split() if t[0] != '<' and t[-1] != '>'])
    print(f'Adding {len(normal_words)} normal tokens')
    tokenizer.add_tokens([AddedToken(t, single_word=True, lstrip=True, rstrip=True, normalized=False) for t in normal_words])
    #tokenizer.resize_token_embeddings(len(specials.k) + len(normal_words))
    print('Done')

    return tokenizer