from tqdm.auto import tqdm
import torch

from datasets import PairLoader
from models import CycleGAN

if __name__ == '__main__':

    dir_X = './monet2photo/trainB'
    dir_Y = './monet2photo/trainA'
    image_size = 256
    batch_size = 1
    image_channels = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5

    dataloader = PairLoader(dir_X, dir_Y, image_size, batch_size)
    model = CycleGAN(image_channels, image_channels).to(device)

    for epoch in range(epochs):

        ep_loss_G, ep_loss_D = 0, 0
        for image_X, image_Y in tqdm(dataloader, leave=False):

            image_X = image_X.to(device)
            image_Y = image_Y.to(device)

            loss_G, loss_D = model.train(image_X, image_Y)
            ep_loss_G += loss_G.item()
            ep_loss_D += loss_D.item()

        ep_loss_G /= len(dataloader)
        ep_loss_D /= len(dataloader)
        print(f'Epoch: {epoch+1} | Generator Loss:{ep_loss_G} | Discriminator Loss:{ep_loss_D}')
