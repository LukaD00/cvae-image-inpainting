import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import RandomErasing
from torchvision.transforms.functional import erase

class Inpainting():

    def __init__(self, image, mask):
        self.image = image.squeeze(0)
        self.mask = mask

    def inpaint(self, generated):
        result = self.image * (1 - self.mask) + generated.squeeze(0) * self.mask
        return result
    

def read_mask_from_image(image):
    image = image.squeeze(0)
    _, H, W = image.shape
    mask = torch.zeros((H,W)).to(image.device)

    for h in range(H):
        for w in range(W):
            if image[0][h][w]==0 and image[1][h][w]==0 and image[2][h][w]==0:
                mask[h][w] = 1
    
    return mask

class DeleteRandomRectangle(nn.Module):

    def __init__(self):
        super(DeleteRandomRectangle, self).__init__()
        self.random_erasing = RandomErasing(p=1, scale=(0.02, 0.05), inplace=False)

    def forward(self, x):
        x = self.random_erasing(x)
        mask = (x == 0).type(torch.int8)
        return x, mask


class DeleteRectangle(nn.Module):

    def __init__(self, i, j, h, w):
        super(DeleteRectangle, self).__init__()
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def forward(self, x):
        x = x.squeeze(0)
        x = erase(x, self.i, self.j, self.h, self.w, v = 0)
        x = x.unsqueeze(0)
        mask = (x == 0).type(torch.int8)
        return x, mask

class DeleteRectangleBatch(nn.Module):

    def __init__(self, i, j, h, w):
        super(DeleteRectangleBatch, self).__init__()
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def forward(self, x):
        x = erase(x, self.i, self.j, self.h, self.w, v = 0)
        mask = (x == 0).type(torch.int8)
        return x, mask


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

    