import torch
import torchvision.transforms as transforms
import torchvision.utils as utils

from tqdm.auto import tqdm
import os.path
import argparse

from models import CycleGAN
from datasets import TestDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--out', type=str, default='./')
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(args.model, map_location=device)

    ch = state_dict['ch']
    size = state_dict['size']

    dataset = TestDataset(args.src, size)

    model = CycleGAN(ch, ch).to(device)
    model.load_state_dict(state_dict['model'])

    for image, path in tqdm(dataset):

        pred = model.G_X(image.unsqueeze(0).to(device)).cpu().detach()
        out = dataset.inv_norm(pred[0])
        utils.save_image(out, os.path.join(args.out, f'result_{path}'))