import torch
import torch.nn as nn
import torchvision
import tqdm

from datasets import celeba
from models.cvae import cVAE
from models.inpaint import DeleteRandomRectangle, Inpaint
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="/home/rjurisic/Desktop/FER/DUBUCE/runs/finetune")

BCE_loss = nn.BCELoss()
delete_rectangle = DeleteRandomRectangle()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

def crop(x, low, high):
    x[x <= low] = low
    x[x >= high] = high
    return x


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: crop(x, 0., 1.)),
    torchvision.transforms.Resize((109, 89), antialias=True)  # (3, 218, 178) -> (3, 109, 89)
])

train_data = celeba.CelebA(root='/home/rjurisic/Desktop/FER/DUBUCE', download=False, transform=transform)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main_convolution = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 512, kernel_size=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # nn.Conv2d(512, 1024, kernel_size=3),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),
        )

        self.main_dense = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.main_convolution(x)
        maps = torch.mean(features, (-1, -2))

        return self.main_dense(maps)


def train_discriminator(batch, attr, discriminator, generator, optimizer, global_step):
    discriminator.train()
    generator.eval()

    with torch.no_grad():
        batch_size = batch.size(0)
        cropped, mask = delete_rectangle(batch)
        generated, _, _ = generator(cropped, mask, attr)

    generated_loss = BCE_loss(discriminator(generated).squeeze(), torch.zeros(batch_size, device=device))
    real_loss = BCE_loss(discriminator(batch).squeeze(), torch.ones(batch_size, device=device))
    total_loss = (generated_loss + real_loss) / 2

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    writer.add_scalar("discriminator_loss", total_loss.cpu().item(), global_step=global_step)


def train_generator(batch, attr, discriminator, generator, optimizer, global_step):
    discriminator.eval()
    generator.train()

    cropped, mask = delete_rectangle(batch)
    generated, mean, logvar = generator(cropped, mask, attr)

    discriminator_loss = BCE_loss(discriminator(generated).squeeze(), torch.ones(batch.size(0), device=device))
    KL_divergence_loss = torch.mean(-1 - logvar + torch.exp(logvar) + mean ** 2)
    reconstruction_loss = BCE_loss(generated, batch)

    total_loss = 4 * reconstruction_loss + 0.5 * KL_divergence_loss + discriminator_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    writer.add_scalar("generator_loss", total_loss.cpu().item(), global_step=global_step)


def save_models(generator, discriminator):
    torch.save({"net": discriminator.state_dict()}, "discriminator.pt")
    torch.save({"net": generator.state_dict()},"generator_fineTuned.pt")


def main_train_like_a_gan():
    cvae = cVAE((3, 109, 89), 2, nhid=64, ncond=8)
    checkpoint = torch.load("../cVAE.pt", map_location=device)
    cvae.load_state_dict(checkpoint["net"])
    cvae.to(device)

    discriminator = Discriminator().to(device)
    generator = Inpaint(cvae).to(device)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01, weight_decay=0.0001)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.01, weight_decay=0.0001)

    epochs = 1
    iteration = 0
    discriminator_warmup_iterations = len(train_data) / batch_size * 0.05
    for epoch in range(epochs):
        for X, y in tqdm.tqdm(train_iter, ncols=50):
            iteration += 1

            X = X.to(device)
            y = y.to(device)

            train_discriminator(X, y, discriminator, generator, discriminator_optimizer, iteration)

            if iteration < discriminator_warmup_iterations:
                continue

            train_generator(X, y, discriminator, generator, generator_optimizer, iteration)

    save_models(generator, discriminator)
    writer.close()


if __name__ == '__main__':
    main_train_like_a_gan()
    # print(sum(p.numel() for p in Discriminator().parameters()))
