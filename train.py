import torch
import torchvision
import os, time, tqdm
from models.cvae import loss, cVAE
from utils import EarlyStop
from datasets import celeba
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="/home/rjurisic/Desktop/FER/DUBUCE/runs/train_baseline/short/")

############## loading data ###################

def crop(x, low, high):
    x[x <= low] = low
    x[x >= high] = high
    return x


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: crop(x, 0., 1.)),
    torchvision.transforms.Resize((109, 89), antialias=True),  # (3, 218, 178) -> (3, 109, 89)
    torchvision.transforms.CenterCrop((64, 64)),
])

train_data = celeba.CelebA(root='/home/rjurisic/Desktop/FER/DUBUCE', download=False, transform=transform)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

############## loading models ###################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = cVAE((3, 64, 64), 2, nhid=100, ncond=16)
net.to(device)
# print(net)
save_name = "cVAE.pt"

############### training #########################

lr = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0001)


def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if not (ans == 'y'):
        checkpoint = torch.load(save_name, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr

max_epochs = 1
early_stop = EarlyStop(patience=20, save_name=save_name)
net = net.to(device)

print("training on ", device)
iteration = 0
for epoch in range(max_epochs):

    train_loss, n, start = 0.0, 0, time.time()
    for X, y in tqdm.tqdm(train_iter, ncols=50):
        X = X.to(device)
        y = y.to(device)
        X_hat, mean, logvar = net(X, y)

        l = loss(X, X_hat, mean, logvar).to(device)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        n += 1

        writer.add_scalar('train_loss', l.cpu().item(), global_step=iteration)
        iteration += 1

    train_loss /= n
    print('epoch %d, train loss %.4f , time %.1f sec' % (epoch, train_loss, time.time() - start))

    adjust_lr(optimizer)

    if (early_stop(train_loss, net, optimizer)):
        break

writer.close()
checkpoint = torch.load(early_stop.save_name)
net.load_state_dict(checkpoint["net"])
