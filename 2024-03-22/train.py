import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from data import IMDBDataset
from net import SentenceClassifier

parser = argparse.ArgumentParser()

# 인자 추가
parser.add_argument("--data_path", type=str, default="../../datasets/IMDB/")
parser.add_argument("--ratio", type=float, default=0.2)

parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

parser.add_argument("--model", type=str, default='gru')
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bidirectional", type=bool, default=False)

parser.add_argument("--optimizer", type=str, default='Adam')
parser.add_argument("--lr_scheduler", type=str, default='Step')
parser.add_argument("--step_size", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)

parser.add_argument("--save", type=str, default="rnn_1")

# loss funtion 시각화 함수
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train_Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation_Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f"{args.save}"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

# batch 내 텍스트 데이터 길이 패딩
def collate_fn(batch):
    text, label = zip(*batch)
    padded_text = pad_sequence(text, batch_first=True, padding_value=0)
    label = torch.tensor(label, dtype=torch.long)
    return padded_text, label

# optimizer 설정
def get_optimizer():
    if args.optimizer == 'SGD':
        return SGD
    elif args.optimizer == 'Adam':
        return Adam
    else:
        raise ValueError(args.optimizer)

# 스케줄러 타입 설정
def get_scheduler():
    if args.lr_scheduler == "Step":
        return StepLR
    elif args.lr_scheduler == "Cosine":
        return CosineAnnealingLR
    else:
        raise ValueError(args.lr_scheduler)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMDBDataset을 로딩, 데이터셋을 분할
    dataset = IMDBDataset(args.data_path)
    train_dataset, val_dataset = dataset.split_dataset(ratio=args.ratio)
    vocab_size = len(dataset.vocab)
    
    # DDP 설정
    if args.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        rank = 0
    
    # 데이터 불러오기
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                          shuffle=(not args.is_ddp), num_workers=4, pin_memory=True, collate_fn=collate_fn), \
                               DataLoader(val_dataset, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 모델 불러오기
    net = SentenceClassifier(vocab_size, hidden_size=args.hidden_size, embed_size=args.embed_size, n_layers=args.n_layers, 
                             dropout=args.dropout, bidirectional=args.bidirectional, model_type=args.model).to(device)
    if args.is_ddp:
        net = DistributedDataParallel(net)
    
    # 크로스 엔트로피 손실 함수 사용
    criterion = CrossEntropyLoss()
    
    # 옵티마이저
    optimizer_type = get_optimizer()
    optimizer = optimizer_type(net.parameters(), lr=args.learning_rate)

    # 스케줄러
    scheduler_type = get_scheduler()
    try:
        scheduler = scheduler_type(optimizer, step_size=args.step_size, gamma=args.gamma)
    except:
        scheduler = scheduler_type(optimizer, T_max=args.step_size)

# AMP 사용 여부에 따라 스케일러 초기화
if args.is_amp:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# loss 저장 리스트
train_losses = []
val_losses = []
# loss 초기화
best_loss = float('inf')

for epoch in range(args.epoch):
    # DDP 사용 시 에포크마다 샘플러 설정
    if args.is_ddp:
        train_sampler.set_epoch(epoch)

    # 훈련 초기화
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # 훈련 모드
    net.train()

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 옵티마이저 그라디언트 초기화
        optimizer.zero_grad()

        # 스케일러 사용 시
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_loss += loss.item()
    
    # 스케줄러 업데이트
    scheduler.step()
    
    # 평균 훈련 손실 및 정확도 계산
    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(train_loss)

    # 검증 손실 및 정확도 초기화
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # 모델을 평가 모드로 설정
    net.eval()

    # 검증 데이터 로더를 이용한 반복, 그라디언트 계산 없음
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    # 평균 검증 손실 및 정확도 계산
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss)

    # 랭크 0(주로 메인 프로세스)에서만 결과 출력 및 최적 모델 저장
    if rank == 0:
        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # 검증 손실이 개선되었는지 확인 후 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            # 분산 데이터 병렬 처리 사용 시 모델의 상태 사전 가져오기 조정
            if args.is_ddp:
                weights = net.module.state_dict()
            else:
                weights = net.state_dict()
                torch.save(weights, f'./result/{args.save}.pth')

    if rank == 0:
        plot_loss(train_losses, val_losses)

    if args.is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    main()
