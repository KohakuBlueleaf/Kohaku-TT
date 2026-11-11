# SDXL on tt

The SDXL UNet and VAE decoder as plain PyTorch modules, loading the weights of
[Kohaku-XL-Zeta](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta) from the local Hugging
Face cache (`--model` takes any SDXL repo in the diffusers layout). No model library is
used for compute; the state-dict names follow diffusers, so
`load_state_dict(strict=True)` accepts the checkpoint as it is.

```bash
cd examples/07_sdxl
# CPU: strict load, then fp32 outputs against diffusers (needs `diffusers`)
python check_reference.py --latent 64 --out ref_64.pt
# one chip: bf16 forward, time and error against the fp32 CPU output
python run.py --ref ref_64.pt --latent 64
```

`check_reference.py` prints, for the UNet and the VAE decoder, `strict load ok` and the
relative L2 error of these modules against diffusers' own (0 on the CPU: the same ops in
the same order). `run.py` prints one line per model: first-call time (program builds),
median of three more calls, the error against the saved fp32 CPU output and the output
shape. Without `--ref` it prints only the times; `--only unet|vae` runs one of them.

`--latent` is the latent side (64 gives a 512 x 512 image, 128 a 1024 x 1024 one). Text
conditioning and latents are seeded random; the text encoders and the sampler loop are
not part of this example (`06_moss_image/` has a full pipeline).

| file | contents |
|---|---|
| `unet.py` | UNet: resnets, spatial transformers (self and cross attention, GEGLU), down / up stages with skips, time and added-condition embeddings |
| `vae.py` | VAE decoder: resnets, a single-head attention mid block, nearest upsampling |
| `check_reference.py` | strict load and parity with diffusers on the CPU; saves the fp32 outputs |
| `run.py` | the chip run |

Everything runs in the weight dtype (bf16 on the chip), including GroupNorm, LayerNorm and
softmax. The fp32 CPU reference of the UNet needs about 20 GB of host memory.
