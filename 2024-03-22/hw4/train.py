import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import TextDataset, split_dataset
from lstm import Seq2Seq

parser = argparse.ArgumentParser()
# -- hyperparameter about data
parser.add_argument("--data_path", type=str, default="../../datasets/fra_eng.txt")
parser.add_argument("--num_sample", type=int, default=33000)
parser.add_argument("--max_seq_len", type=int, default=16)

# -- hyperparameter about ddp &amp
parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

# --hyperparameter about model
parser.add_argument("--hidden_size", type=int, default=256)

# --hyperparameter about train
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epoch", type=int, default=20)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = f"LSTM Seq2Seq Model"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

def train(args):
    os.makedirs("result", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    text_dataset = TextDataset(args.data_path, args.num_sample, args.max_seq_len)
    train_dataset, val_dataset = split_dataset(text_dataset)
    
    # set the ddp
    if args.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        rank = 0
    
    # define the dataloader
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                          shuffle=(not args.is_ddp), num_workers=4, pin_memory=True), \
                               DataLoader(val_dataset, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=4, pin_memory=True)
    
    src_vocab_size, trg_vocab_size = len(text_dataset.src_vocab), len(text_dataset.trg_vocab)
    net = Seq2Seq(src_vocab_size, trg_vocab_size, args.hidden_size).to(device)
    if args.is_ddp:
        net = DistributedDataParallel(net)
    
    # define the loss_function & optimizer
    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    # define the scaler
    if args.is_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epoch):
        if args.is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        for data in tqdm(train_loader):
            encoder_inputs = data["encoder_input"].to(device)
            decoder_inputs = data["decoder_input"].to(device)
            decoder_targets = data["decoder_target"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = net(encoder_inputs, decoder_inputs)
                    loss = criterion(output.view(-1, output.size(-1)), decoder_targets.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = net(encoder_inputs, decoder_inputs)
                loss = criterion(output.view(-1, output.size(-1)), decoder_targets.view(-1))

                loss.backward()
                optimizer.step()

            # calculate accuracy
            mask = decoder_targets != 0
            train_correct += ((output.argmax(dim=-1) == decoder_targets) * mask).sum().item()
            train_total += mask.sum().item()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        net.eval()

        with torch.no_grad():
            for data in tqdm(val_loader):
                encoder_inputs = data["encoder_input"].to(device)
                decoder_inputs = data["decoder_input"].to(device)
                decoder_targets = data["decoder_target"].to(device)

                output = net(encoder_inputs, decoder_inputs)
                loss = criterion(output.view(-1, output.size(-1)), decoder_targets.view(-1))

                # calculate accuracy
                mask = decoder_targets != 0
                val_correct += ((output.argmax(dim=-1) == decoder_targets) * mask).sum().item()
                val_total += mask.sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epoch}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_loss < best_loss:
                best_loss = val_loss
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
    args = parser.parse_args()
    train(args)