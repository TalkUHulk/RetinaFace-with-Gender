import torch
import sys

sys.path.append("../scripts")
sys.path.append("..")

from models.retinaface import RetinaFace
from data import cfg_mnetv2, cfg_re50

# cfg = cfg_re50
# weight_path = '../ckpt/Resnet50_Gender_Final.pth'
# onnx_dst = "../ckpt/retinaface-res50-gender-320.onnx"

cfg = cfg_mnetv2
weight_path = '../ckpt/MobileNet_v2_Final.pth'
onnx_dst = "../ckpt/retinaface-mbv2-320.onnx"


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch_model = RetinaFace(cfg=cfg)
torch_model = load_model(torch_model, weight_path, True)
torch_model.eval()

# set the model to inference mode


# Input to the model
x = torch.randn(1, 3, 320, 320)
torch_out = torch_model(x)

# Export the model
# torch.onnx.export(torch_model,  # model being run
#                   x,  # model input (or a tuple for multiple inputs)
#                   onnx_dst,  # where to save the model (can be a file or file-like object)
#                   export_params=True,  # store the trained parameter weights inside the model file
#                   opset_version=11,  # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],  # the model's input names
#                   output_names=["bbox", "cls", "ldm", "gender"],  # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
#                                 'bbox': {0: 'batch_size'},
#                                 'cls': {0: 'batch_size'},
#                                 'ldm': {0: 'batch_size'},
#                                 'gender': {0: 'batch_size'}})


torch.onnx.export(torch_model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  onnx_dst,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=["bbox", "cls", "ldm"],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'bbox': {0: 'batch_size'},
                                'cls': {0: 'batch_size'},
                                'ldm': {0: 'batch_size'}})
