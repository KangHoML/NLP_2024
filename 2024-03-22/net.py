import torch
import torch.nn as nn

'''
Input:
    input sequence of text (batch_size, seq_len)
Output:
    binary class(0 or 1) of sentence (batch_size, 2)
'''

class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_size, embed_size, n_layers=1,
                 dropout=0, bidirectional=False, model_type='lstm'):
        super().__init__()
        
        self.hidden_size = hidden_size  # 은닉층 크기
        self.n_layers = n_layers  # RNN 레이어의 수
        self.bidirectional = bidirectional  # 양방향 RNN 여부
        self.model_type = model_type  # 모델 유형 ('rnn', 'lstm', 'gru')
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)

        # 모델 유형에 따른 컨텍스트 레이어 설정(rnn, lstm, gru)
        if model_type == "rnn":
            self.context = nn.RNN(embed_size, hidden_size, num_layers=n_layers, 
                                  bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "lstm":
            self.context = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, 
                                   bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "gru":
            self.context = nn.GRU(embed_size, hidden_size, num_layers=n_layers, 
                                  bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        # 양방향 여부에 따른 분류기 레이어 설정
        if bidirectional:
            self.classifier = nn.Linear(hidden_size * 2, 2)  # 양방향일 경우 크기 2배
        else:
            self.classifier = nn.Linear(hidden_size, 2)  # 단방향일 경우
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)

        # 가중치 초기화 함수 호출
        self._init_weights()

    def forward(self, x):
        embedded = self.embedding(x)  # 입력을 임베딩
        h_0 = self._init_state(batch_size=embedded.size(0))  # 초기 상태 설정

        output, _ = self.context(embedded, h_0)  # 컨텍스트 레이어 통과
        output = output[:, -1, :]  # 시퀀스의 마지막 출력 선택
        
        output = self.dropout(output)  # 드롭아웃 적용
        logits = self.classifier(output)  # 분류 레이어 통과
        
        return logits
    
    def _init_state(self, batch_size=1):
        # 초기 상태 설정
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1

        h_0 = weight.new(self.n_layers * num_directions, batch_size, self.hidden_size).zero_()
        c_0 = weight.new(self.n_layers * num_directions, batch_size, self.hidden_size).zero_()

        if self.model_type == 'lstm':
            return (h_0, c_0)  # LSTM은 (h_0, c_0)을 반환
        else:
            return h_0  # RNN과 GRU는 h_0만 반환
    
# 가중치 초기화 함수
def _init_weights(self):
    for name, param in self.named_parameters():
        if 'weight_ih' in name:  # weight_ih는 Xavier Uniform 사용
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:  # weight_hh는 Orthogonal 사용
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

    nn.init.xavier_uniform_(self.embedding.weight)
    nn.init.xavier_uniform_(self.classifier.weight)
    self.classifier.bias.data.fill_(0)

if __name__ == "__main__":
    src_vocab_size = 40710
    

    net = SentenceClassifier(src_vocab_size, hidden_size=64, embed_size=128, n_layers=2, model_type='rnn')
    random_input = torch.randint(low=0, high=src_vocab_size, size=(64, 500), dtype=torch.long)

    random_output = net(random_input)
    # 출력 형태를 출력, 예측 결과의 차원을 확인할 수 있음
    print(f'Output Shape: {random_output.shape}')
