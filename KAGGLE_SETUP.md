# PersonaLive on Kaggle

This repository can be run on a Kaggle GPU notebook while keeping the large pretrained weights outside GitHub.

## Tested setup

- Kaggle Tesla T4 GPU
- Python 3.10 environment
- NumPy 1.26.3
- PyTorch 2.1.0 + CUDA 12.1
- xFormers 0.0.22.post7

PersonaLive's pinned Python dependencies are listed in `requirements_base.txt`.

## Persistent weights

The pretrained weights are intentionally **not committed to GitHub**. In the current Kaggle setup they are stored in a private Kaggle Dataset and mounted read-only into the notebook.

Expected repository path:

```text
/kaggle/working/PersonaLive/pretrained_weights
```

If the weights dataset is mounted at:

```text
/kaggle/input/datasets/alielnaggar06/personalive-pretrained-weights
```

connect it with:

```bash
rm -rf /kaggle/working/PersonaLive/pretrained_weights
ln -s /kaggle/input/datasets/alielnaggar06/personalive-pretrained-weights \
      /kaggle/working/PersonaLive/pretrained_weights
```

Expected weight directories include `personalive`, `sd-image-variations-diffusers`, `sd-vae-ft-mse`, `onnx`, and `tensorrt`.

## Environment

Use Python 3.10 and install the repository's pinned dependencies. The Kaggle environment used during development lives at:

```text
/tmp/personalive-env
```

`/tmp` is temporary, so this environment must be recreated after a fresh Kaggle session starts.

Before inference, verify CUDA:

```bash
/tmp/personalive-env/bin/python -c "import numpy, torch, xformers; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('xFormers:', xformers.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Run with the reusable wrapper

The wrapper automatically reads the driving video's frame count, rounds it down to a multiple of four, creates a unique PersonaLive run name, and locates the clean generated video.

Example:

```bash
/tmp/personalive-env/bin/python run_personalive.py \
  --python /tmp/personalive-env/bin/python \
  --reference-image /path/to/reference.png \
  --driving-video /path/to/driving.mp4 \
  --output /kaggle/working/PersonaLive/final_outputs/avatar.mp4
```

PersonaLive's `inference_offline.py` saves videos at 25 FPS. The wrapper currently preserves that native output rather than retiming it.

## Important

Do not commit model weights, generated results, personal reference images, driving videos, API keys, or `.env` files to this repository.
