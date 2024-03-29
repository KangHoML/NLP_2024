import os
import torch
import argparse

from collections import OrderedDict
from torch.utils.data import DataLoader

from data import IMDBDataset
from net import SentenceClassifier  # 이미 정의된 모델 클래스 import
from train import collate_fn

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()

# 인자 추가
parser.add_argument("--data_path", type=str, default="../../datasets/IMDB/")
parser.add_argument("--weight_path", type=str, default="../../pth/IMDB/")
parser.add_argument("--pth_name", type=str, default="lstm_ddp")

parser.add_argument("--model", type=str, default='gru')
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bidirectional", type=bool, default=False)

parser.add_argument("--batch_size", type=int, default=32)

def main():
    # GPU 사용 가능 여부에 따라 device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 테스트 데이터셋 로딩
    test_dataset = IMDBDataset(root=args.data_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 모델 가중치 파일 경로 구성
    weight_path = os.path.join(args.weight_path, args.pth_name + '.pth')
    
    # 가중치 로드
    state_dict = torch.load(weight_path, map_location=device)
    
    # 모델 인스턴스 생성을 위한 어휘 크기 추출
    vocab_size = state_dict['embedding.weight'].size()[0]
    
    # 모델 인스턴스 생성 및 device로 이동
    net = SentenceClassifier(vocab_size, hidden_size=args.hidden_size, embed_size=args.embed_size, n_layers=args.n_layers, 
                             dropout=args.dropout, bidirectional=args.bidirectional, model_type=args.model).to(device)

    # DataParallel로 저장된 가중치의 'module.' 접두사를 제거
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 'module.' 제거
        new_state_dict[name] = v

    # 가중치를 모델에 로드
    net.load_state_dict(new_state_dict)
    net = net.to(device)

    # 모델을 평가 모드로 설정
    net.eval()

    # 정확도 계산을 위한 변수 초기화
    correct = 0
    total = 0
    
    # 그래디언트 계산을 방지하기 위해 no_grad 컨텍스트 사용
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 device로 이동
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 최대 확률을 가진 인덱스 추출
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 정답과 비교하여 맞힌 개수 계산
    
    # 정확도 출력
    accuracy = 100 * correct / total
    print(f'{args.pth_name} Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    global args
    args = parser.parse_args()  # 인자값을 파싱

    main()  # 메인 함수 실행
