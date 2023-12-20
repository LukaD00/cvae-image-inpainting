import torch

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

    