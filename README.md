# Orpheus TTS Engine (Windows + llama.cpp)

This repository shares experimental performance gains for [Canopy Labs Orpheus](https://github.com/canopyai/Orpheus-TTS)

Runs on CUDA 12.4 and Python 3.12.9.

You also need to download:

- [Multi-Scale Neural Audio Codec (SNAC) safetensors](https://huggingface.co/Annuvin/snac-24khz-ST/tree/main)
- [Orpheus 3B 0.1 Finetune Q4_K_M GGUF](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF/tree/main) OR [Orpheus 3B 0.1 Finetune safetensors](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/tree/main)

```bash
# Recommended: Create a venv and activate it
python -m venv env
venv\Scripts\activate 
# Install torch 2.6 if you don't already have it
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install the requirements
pip3 install -r requirements.txt
# Run the test
python test.py --model-path path/to/model.gguf --snac-path path/to/snac-folder --output-path path/to-outputs
```

For arguments, see [test.py](./test.py)

## License

Inherited from Canopy Labs Orpheus TTS.

[Apache-2.0 License](https://github.com/canopyai/Orpheus-TTS?tab=Apache-2.0-1-ov-file)