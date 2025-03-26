from orpheus_model_llamacpp import OrpheusModelLlamaCpp
from orpheus_model_transformers import OrpheusModelTransformers
from snac_model_transformers import SnacModel

def create_orpheus_engine(model_path: str, snac_path: str):
    audio_model = SnacModel(snac_path)
    if model_path.endswith(".gguf"):
        return OrpheusModelLlamaCpp(model_path, audio_model)
    else:
        return OrpheusModelTransformers(model_path, audio_model)

