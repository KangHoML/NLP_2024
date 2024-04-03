from transformers import T5Tokenizer
tokenizer = T5Tokenizer(vocab_file="t5-sp-bpe-nsmc-byte-fallback.model")
tokenizer.save_pretrained("t5-tokenizer-bpe-nsmc-byte-fallback")
lines = [
  "`DEVOCEAN`은 SK그룹의 대표 개발자 커뮤니티이자🧑",
  "내/외부 개발자 간 소통과 성장을 위한 플랫폼을 상징합니다.👋",
  "`Developers`' Ocean 개발자들을 위한 영감의 바다🙏",
  "`Devotion` 헌신,몰두,전념💯",
  "`Technology for Everyone` 모두를 위한 기술👍"
  ]

for line in lines:
    tokens = tokenizer.tokenize(line)
    inputs = tokenizer(line)    
    decoded_sequence = tokenizer.decode(inputs['input_ids'])
    print(line)
    print(tokens)    
    print(decoded_sequence)
    print()
