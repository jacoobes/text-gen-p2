import sentencepiece as spm

tokenizer_location = "bptokenizer"
def train_bpe(response, output: str):
    spm.SentencePieceTrainer.Train(
      input=response, 
      model_prefix=output,
      vocab_size=10_000,
      model_type='bpe',
      control_symbols=",".join(['<bos>', '<eos>', '<pad>' ]),
      bos_id=1,
      eos_id=2,
      pad_id=3
    )
      

train_bpe('./corpus.txt', output=tokenizer_location)
