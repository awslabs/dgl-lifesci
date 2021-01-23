## Training

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py
CUDA_VISIBLE_DEVICES=0 python vaetrain.py -m pre_model/model.iter-2
```

## Test

```bash
CUDA_VISIBLE_DEVICES=0 python reconstruct.py -m vae_model/model.iter-4
```
