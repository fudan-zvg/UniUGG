import torch
import os
weight_path = "checkpoints"
model_path = "src/encoder/RADIO"

def get_radio_model():
    torch.hub.set_dir(weight_path)
    model_version = os.path.join(weight_path, "radio-v2.5-l_half.pth.tar")
    radio_model = torch.hub.load(model_path, 'radio_model',source='local', version=model_version, progress=True, skip_validation=True)
    return radio_model

def get_radio_model_cilp():
    model_version = os.path.join(weight_path, "radio-v2.5-l_half.pth.tar")
    radio_model = torch.hub.load(model_path, 'radio_model',source='local', version=model_version, adaptor_names='clip', progress=True, skip_validation=True)
    return radio_model