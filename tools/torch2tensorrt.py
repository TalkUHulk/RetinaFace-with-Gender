import torch
from torch2trt.torch2trt import *
from torch2trt import torch2trt
from torch2trt import tensorrt_converter
import sys

sys.path.append("../scripts")
sys.path.append("..")
from models.retinaface import RetinaFace
from data import cfg_mnetv2, cfg_re50


cfg = cfg_mnetv2
weight_path = './ckpt/MobileNet_v2_Final.pth'
onnx_dst = "./ckpt/retinaface-mbv2-320.onnx"

# cfg = cfg_re50
# weight_path = './ckpt/Resnet50_Gender_Final.pth'
# onnx_dst = "./ckpt/retinaface-res50-gender-320.onnx"


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


@tensorrt_converter('torch.softmax')
def convert_softmax(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    # get dims from args or kwargs
    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    elif len(ctx.method_args) >= 2:
        dim = ctx.method_args[1]

    # convert negative dims
    #     import pdb
    #     pdb.set_trace()
    if dim < 0:
        dim = len(input.shape) + dim

    axes = 1 << (dim - 1)

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)

torch_model = RetinaFace(cfg=cfg)
torch_model = load_model(torch_model, weight_path, False)
torch_model.eval().cuda()

# set the model to inference mode

# Input to the model
x = torch.randn(1, 3, 320, 320).cuda()

model_trt = torch2trt(torch_model, [x])

y = torch_model(x)
y_trt = model_trt(x)

print(y[0].shape)
# check the output against PyTorch
print(torch.max(torch.abs(y[0] - y_trt[0])))

torch.save(model_trt.state_dict(), weight_path.replace(".pth", "_Trt.pth"))

