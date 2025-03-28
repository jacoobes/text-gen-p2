import io
import glob
import os
import sentencepiece as spm

tokenizer_location = "bptokenizer"
def train_bpe(response, output: str):
    spm.SentencePieceTrainer.Train(
      input=response, 
      model_prefix=output,
      vocab_size=10000,
      model_type='bpe',
      bos_id=1,
      eos_id=2,
      pad_id=3,
      user_defined_symbols=",".join(['<bos>', '<eos>', '<pad>' ])
    )
      

if not os.path.exists(tokenizer_location):
    corpus = ""
    for file in glob.glob("*.txt", root_dir="./raw"):
        with open('raw/'+file, 'r', encoding="utf8") as f:
            corpus +=  f.read() + " "
    with open('./corpus.txt', 'w+', encoding='utf8') as cr:
        cr.write(corpus)

    train_bpe('corpus.txt', output=tokenizer_location)
