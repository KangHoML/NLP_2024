import os
import torch
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import CIFAR10Dataset
from resnet import ResNet

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./datasets/CIFAR-10")
parser.add_argument("--model", type=str, default="VGG16")
parser.add_argument("--bn_flag", type=bool, default=True)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default=20)

def get_model(model):
    if "ResNet" in model:
        return ResNet(cfg=model)
    else:
        raise NotImplementedError(model)

def train(args):
    os.makedirs("result", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    train_dataset, val_dataset = CIFAR10Dataset(args.data_path, True), \
                                 CIFAR10Dataset(args.data_path, False)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    net = get_model(args.model).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(args.epoch):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        net.eval()

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

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)




    
    