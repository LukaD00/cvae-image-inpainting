import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import RandomErasing


class DeleteRandomRectangle(nn.Module):

    def __init__(self):
        super(DeleteRandomRectangle, self).__init__()
        self.random_erasing = RandomErasing(p=1, scale=(0.02, 0.2), inplace=False)

    def forward(self, x):
        x = self.random_erasing(x)
        mask = (x == 0).type(torch.int8)
        return x, mask


class Inpaint(nn.Module):

    def __init__(self, cvae):
        super(Inpaint, self).__init__()
        self.cvae = cvae

    def forward(self, x, mask, attr):
        generated, mean, logvar = self.cvae(x, attr)

        return generated * mask + x * (1 - mask), mean, logvar


if __name__ == '__main__':
    randomErase = torchvision.transforms.RandomErasing(p=1)

    def crop(x, low, high):
        x[x <= low] = low
        x[x >= high] = high
        return x

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: crop(x, 0., 1.)),
        torchvision.transforms.Resize((109, 89), antialias=True)  # (3, 218, 178) -> (3, 109, 89)
    ])

    img = Image.open("../test_pics/000118_removed_lips.png")
    img = transform(img)[0:3, : , :]
    img = randomErase(img)
    img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    plt.imshow(img)
    plt.show()
