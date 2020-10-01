from tqdm.auto import tqdm
import argparse
import torch

from datasets import PairLoader
from models import CycleGAN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_X', type=str, required=True)
    parser.add_argument('--dir_Y', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = PairLoader(args.dir_X, args.dir_Y, args.image_size, args.batch)
    model = CycleGAN(args.channels, args.channels).to(device)

    for epoch in range(args.epochs):

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

    if args.save_path is not None:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'size': args.image_size,
            'ch': args.channels
        }
        torch.save(state_dict, args.save_path)