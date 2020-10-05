# cyclegan-pytorch
Pytorch implementation of Cycle GAN.

## Usage
Train
```bash
python train.py --dir_X=trainB --dir_Y=trainA --epochs=25 --save_path=models/model.ckpt
```
Test
```bash
python test.py --model=models/model.ckpt --src=testB --out=out
```

## Result
Sample results are on images folder. I used train data from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)