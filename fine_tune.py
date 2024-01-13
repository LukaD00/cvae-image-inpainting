import torch
import torch.nn as nn
import torchvision
import tqdm

from datasets import celeba
from models.cvae import cVAE
from datasets.inpainting import DeleteRandomRectangle, DeleteSmilingRectangle, DeleteRandomBigRectangle
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

BCE_loss = nn.BCELoss(reduction='sum')

#delete_rectangle = DeleteRandomRectangle()
delete_rectangle = DeleteRandomBigRectangle()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

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

train_data = celeba.CelebA(root='C:/Datasets', download=False, transform=transform, target_attributes=None)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)

        return out.squeeze()


def train_discriminator(batch, batch_cropped, attr, discriminator, generator, optimizer, global_step):
    with torch.no_grad():
        batch_size = batch.size(0)
        generated, _, _ = generator(batch_cropped, attr)

    loss_item = 0

    optimizer.zero_grad()
    real_loss = BCE_loss(discriminator(batch), torch.ones(batch_size, device=device))
    real_loss.backward()
    optimizer.step()
    loss_item += real_loss.item()

    optimizer.zero_grad()
    generated_loss = BCE_loss(discriminator(generated), torch.zeros(batch_size, device=device))
    generated_loss.backward()
    optimizer.step()
    loss_item += generated_loss.item()

    writer.add_scalar("discriminator_loss", loss_item, global_step=global_step)


def train_generator(batch, batch_cropped, attr, discriminator, generator, optimizer, global_step):
    generated, mean, logvar = generator(batch_cropped, attr)

    discriminator_loss = BCE_loss(discriminator(generated), torch.ones(batch.size(0), device=device))
    KL_divergence_loss = torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    reconstruction_loss = BCE_loss(generated, batch)

    total_loss = 2 * reconstruction_loss + 0.5 * KL_divergence_loss + 10 * discriminator_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    writer.add_scalar("generator_loss", total_loss.cpu().item(), global_step=global_step)


def save_models(generator, discriminator, epoch):
    torch.save({"net": discriminator.state_dict()}, f"./models/weights/discriminator-epoch{epoch}.pt")
    torch.save({"net": generator.state_dict()},f"./models/weights/cVAE2_finetuned-epoch{epoch}.pt")


def main_train_like_a_gan():
    cvae = cVAE((3, 64, 64), 2, nhid=512, ncond=16)
    checkpoint = torch.load("./models/weights/cVAE.pt", map_location=device)
    cvae.load_state_dict(checkpoint["net"])
    cvae.to(device)

    discriminator = Discriminator().to(device)
    generator = cvae.to(device)

    discriminator.train()
    generator.train()

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01, weight_decay=0.0001)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01, weight_decay=0.0001)

    epochs = 100
    iteration = 0
    discriminator_warmup_iterations = len(train_data) / batch_size * 0.05

    for epoch in range(epochs):
        for X, y in tqdm.tqdm(train_iter, ncols=50):
            iteration += 1

            X_cropped, _ = delete_rectangle(X)
            X_cropped = X_cropped.to(device)
            X = X.to(device)
            y = y.to(device)

            train_discriminator(X, X_cropped, y, discriminator, generator, discriminator_optimizer, iteration)

            if iteration < discriminator_warmup_iterations:
                continue

            train_generator(X, X_cropped, y, discriminator, generator, generator_optimizer, iteration)

            if epoch % 10 == 0:
                save_models(generator, discriminator, epoch)

        
        if epoch % 10 == 0:
            save_models(generator, discriminator, epoch)

    save_models(generator, discriminator, epoch)
    writer.close()


if __name__ == '__main__':
    main_train_like_a_gan()
    # print(sum(p.numel() for p in Discriminator().parameters()))
