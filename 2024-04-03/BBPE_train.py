import sentencepiece as spm
from pathlib import Path

data_dir = './data'
paths = [str(x) for x in Path(data_dir).glob("*.txt")]
corpus = ",".join(paths)
prefix = "t5-sp-bpe-nsmc"
vocab_size = 31900-7
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이 (너무 길면 에러발생)
    " --pad_id=0 --pad_piece=<pad>" + # pad (0)
    " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
    " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=</s>" + # end of sequence (3)
    " --user_defined_symbols=<sep>,<cls>,<mask>") # 사용자 정의 토큰

import sentencepiece as spm
from pathlib import Path

data_dir = './data'
paths = [str(x) for x in Path(data_dir).glob("*.txt")]
corpus = ",".join(paths)
prefix = "t5-sp-bpe-nsmc-byte-fallback"
vocab_size = 31900-7
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이 -> 이게 너무 길면 에러발생함
    " --pad_id=0 --pad_piece=<pad>" + # pad (0)
    " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
    " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=</s>" + # end of sequence (3)
    " --byte_fallback=true" + # add byte_fallback for unk tokens
    " --user_defined_symbols=<sep>,<cls>,<mask>") # 사용자 정의 토큰
