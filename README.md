# ComfyUI BitDance Nodes

This package adds BitDance nodes for ComfyUI with a single loader and modular encode/sampler/VAE nodes.

## Nodes

- `BitDance Loader` -> `BITDANCE_MODEL`, `BITDANCE_TEXT_ENCODER`, `BITDANCE_VAE`
- `BitDance Text Encode` -> `positive`, `negative`, `positive_prompt`
- `BitDance Text Encode Cached` -> `positive`, `negative`, `positive_prompt`
- `BitDance Sampler` -> `BITDANCE_LATENT`
- `BitDance VAE Decode` -> `IMAGE`
- `BitDance VAE Encode` -> `BITDANCE_LATENT`

## Install

1. Copy this folder into ComfyUI custom nodes (folder name can be anything), for example:
   `ComfyUI/custom_nodes/Comfyui-bitdance-native`
2. Install dependencies in your ComfyUI Python environment:
   `pip install -r ComfyUI/custom_nodes/Comfyui-bitdance-native/requirements.txt`
3. Restart ComfyUI.

## Model File Locations (3-file setup)

Place your converted files in standard ComfyUI folders:

- Main model -> `ComfyUI/models/diffusion_models/BitDance_14B_MainModel_FP8.safetensors`
- Text encoder -> `ComfyUI/models/text_encoders/BitDance_TextEncoder_FP8.safetensors`
- VAE -> `ComfyUI/models/vae/BitDance_VAE_FP16.safetensors`

Tokenizer files are bundled with this node package, so a separate tokenizer folder in `models/` is not required for local mode.

## Resources

- Hugging Face model files (ComfyUI 3-file package): [comfyuiblog/BitDance-14B-64x-fp8-comfyui](https://huggingface.co/comfyuiblog/BitDance-14B-64x-fp8-comfyui/tree/main)
- Article + workflow guide: [How to fix the generic face bug in BitDance 14B and optimize speed](https://aistudynow.com/how-to-fix-the-generic-face-bug-in-bitdance-14b-optimize-speed/)


## Workflow (Current)

1. Add `BitDance Loader`
2. Add `BitDance Text Encode Cached` (recommended)
3. Add `BitDance Sampler`
4. Add `BitDance VAE Decode`
5. Add ComfyUI `PreviewImage` (local user preview)

Connect:

- `BitDance Loader.bitdance_text_encoder` -> `BitDance Text Encode Cached.text_encoder`
- `BitDance Loader.bitdance_model` -> `BitDance Text Encode Cached.model_to_offload` (optional, for VRAM management)
- `BitDance Text Encode Cached.positive` -> `BitDance Sampler.positive`
- `BitDance Text Encode Cached.negative` -> `BitDance Sampler.negative`
- `BitDance Loader.bitdance_model` -> `BitDance Sampler.model`
- `BitDance Loader.bitdance_vae` -> `BitDance Sampler.vae`
- `BitDance Sampler.bitdance_latent` -> `BitDance VAE Decode.bitdance_latent`
- `BitDance Loader.bitdance_vae` -> `BitDance VAE Decode.vae`
- `BitDance VAE Decode.image` -> `PreviewImage.images`

Starter workflow JSON:

- `workflows/BitDance_Starter.json`

## YouTube

- Youtube Tutorial Guide: https://www.youtube.com/watch?v=4O9ATPbeQyg

## X (Twitter)

- [@aistudynowcom](https://x.com/aistudynowcom)

## Notes

- BitDance is autoregressive and is not a standard UNet denoiser. Use `BitDance Sampler` for generation.
- The loader supports single-file converted checkpoints and bundle-style BitDance layouts.
- FP8 + scale loading is supported for converted files (mixed-precision runtime behavior depends on selected loader mode).
