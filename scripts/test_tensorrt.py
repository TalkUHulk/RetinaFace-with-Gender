from __future__ import print_function
import torch
import argparse
import yaml
from tqdm import tqdm
import os
import numpy as np
import cv2
from torch2trt import TRTModule
import sys

sys.path.append(".")
sys.path.append("..")

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm


def test(_args):
    with open(_args.cfg, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(_args.ckpt))

    confidence_threshold = _args.confidence_threshold
    nms_threshold = _args.nms_threshold
    vis_threshold = _args.vis_threshold
    origin_size = _args.origin_size
    target_size = _args.target_size
    max_size = _args.max_size

    print('Finished loading model!')

    images_path = [x.path for x in os.scandir(_args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

    for i, image_path in tqdm(enumerate(images_path), desc="testing..."):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        if img.shape[0] != _args.target_size or img.shape[1] != _args.target_size:
            img = np.pad(img,
                         ((0, _args.target_size - img.shape[0]), (0, _args.target_size - img.shape[1]), (0, 0)),
                         mode="constant", constant_values=0)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)
        img /= (57, 57, 58)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        y_trt = model_trt(img)

        loc, conf, landms = y_trt[:3]
        if cfg['gender']:
            gender = y_trt[-1]
            gender = gender.cpu()

        loc = loc.cpu()
        conf = conf.cpu()
        landms = landms.cpu()

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        if cfg['gender']:
            genders = gender.squeeze(0).data.cpu().numpy()

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        if cfg['gender']:
            genders = genders[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        if cfg['gender']:
            genders = genders[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        if cfg['gender']:
            genders = genders[keep]
        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        if cfg['gender']:
            dets = np.concatenate((dets, landms, genders * 10000), axis=1).astype(np.float32)
        else:
            dets = np.concatenate((dets, landms), axis=1).astype(np.float32)

        for b in dets:
            if b[4] < vis_threshold:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            if cfg['gender']:
                gender = (b[15] / 10000, b[16] / 10000)
                gender_str = "male:" if gender.index(max(gender)) else "female:"
                gender_str += str(max(gender))

                cv2.putText(img_raw, gender_str, (cx, cy - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))
            # landms
            cv2.circle(img_raw, (b[5], b[6]), 3, (0, 0, 255), 3)
            cv2.circle(img_raw, (b[7], b[8]), 3, (0, 255, 255), 3)
            cv2.circle(img_raw, (b[9], b[10]), 3, (255, 0, 255), 3)
            cv2.circle(img_raw, (b[11], b[12]), 3, (0, 255, 0), 3)
            cv2.circle(img_raw, (b[13], b[14]), 3, (255, 0, 0), 3)

        if not os.path.exists(_args.save):
            os.makedirs(_args.save)
        name = os.path.join(_args.save, image_path.split('/')[-1].split('.')[0] + "_{}_trt.jpg".format(cfg["name"]))
        cv2.imwrite(name, img_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RetinaFace tester.")

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="path to the config yaml file",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.02,
        help="confidence thresh",
    )

    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.04,
        help="nms thresh",
    )

    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=0.5,
        help="visual thresh",
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=320,
        help="target output size",
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=320,
        help="max output size",
    )

    parser.add_argument(
        '--origin_size',
        action="store_true",
        help='keep origin size.')

    parser.add_argument(
        "--images_path", type=str, help="images path to be test."
    )

    parser.add_argument(
        "--save",
        type=str,
        default="./result",
        help="path to save the result",
    )

    args = parser.parse_args()

    test(args)

    print("Test Finished!")

