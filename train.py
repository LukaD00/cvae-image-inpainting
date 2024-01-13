import torch
import torchvision
import os, time, tqdm

from datasets.inpainting import DeleteRandomRectangle, DeleteSmilingRectangle, DeleteRandomBigRectangle
from models.cvae import loss, cVAE
from utils import EarlyStop
from datasets import celeba
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

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

train_data = celeba.CelebA(root='.', download=False, transform=transform, target_attributes="Smiling")
train_iter = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

delete_rectangle = DeleteSmilingRectangle()
#delete_rectangle = DeleteRandomRectangle()
#delete_rectangle = DeleteRandomBigRectangle()

############## loading models ###################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = cVAE((3, 64, 64), 2, nhid=512, ncond=16)
net.to(device)
# print(net)
save_name = "./models/weights/cVAE.pt"

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

max_epochs = 100
early_stop = EarlyStop(patience=20, save_name=save_name, verbose=True)
net = net.to(device)

print("training on ", device)
iteration = 0
for epoch in range(max_epochs):

    train_loss, n, start = 0.0, 0, time.time()
    for X, y in tqdm.tqdm(train_iter, ncols=50):
        X_cropped, _ = delete_rectangle(X)
        X_cropped = X_cropped.to(device)
        X = X.to(device)
        y = y.to(device)
        X_hat, mean, logvar = net(X_cropped, y)

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
