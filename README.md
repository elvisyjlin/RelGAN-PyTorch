# RelGAN-PyTorch

A **PyTorch** implementation of [**RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes**](https://arxiv.org/abs/1908.07269)

The paper is accepted to ICCV 2019. We also have the Keras version [here](https://github.com/willylulu/RelGAN-Keras).

## Get Started

#### Install

1. Python 3.6 or higher
2. PyTorch 0.4.0 or higher
3. All the dependencies

```bash
pip3 install -r requirements.
```

#### Start TensorBoard server

```bash
tensorboard --logdir runs
```

#### Train your RelGAN!

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --data <PATH_TO_CELEBA-HQ> --gpu [--image_size 256]
```

#### Use multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --data <PATH_TO_CELEBA-HQ> --multi_gpu [--image_size 256]
```

#### Specify your own training settings in `config.yaml`

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --config config.yaml
```