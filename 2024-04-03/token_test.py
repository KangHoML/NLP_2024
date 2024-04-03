from transformers import T5Tokenizer
tokenizer = T5Tokenizer(vocab_file="t5-sp-bpe-nsmc-byte-fallback.model")
tokenizer.save_pretrained("t5-tokenizer-bpe-nsmc-byte-fallback")
lines = [
  "`DEVOCEAN`은 SK그룹의 대표 개발자 커뮤니티이자🧑",
  "내/외부 개발자 간 소통과 성장을 위한 플랫폼을 상징합니다.👋",
  "`Developers`' Ocean 개발자들을 위한 영감의 바다🙏",
  "`Devotion` 헌신,몰두,전념💯",
  "`Technology for Everyone` 모두를 위한 기술👍",
  "아 더빙.. 진짜 짜증나네요 목소리",
  "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
  "너무재밓었다그래서보는것을추천한다",
  "교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정",
  "사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다"
  ]

for line in lines:
    tokens = tokenizer.tokenize(line)
    inputs = tokenizer(line)    
    decoded_sequence = tokenizer.decode(inputs['input_ids'])
    print(line)
    print(tokens)    
    print(decoded_sequence)
    print()
