import sys
import argparse

import torch

from torch.nn.functional import cosine_similarity
from torch.linalg import norm

from PIL import Image

from config import cfg
from model import build_model
from dataset import get_trm


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReID custom inference - evaluates a similarity between two"
        " given images."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="path to the inference configuration file"
    )
    parser.add_argument("img_1_path", help="first image file path"),
    parser.add_argument("img_2_path", help="second image file path"),
    parser.add_argument(
        "opts",
        help="overwriting the default configuration",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg, 575)
    param_dict = torch.load(cfg.TEST.WEIGHT)
    model.load_state_dict(param_dict)
    model.cuda()
    model.eval()

    transform = get_trm(cfg, is_train=False)

    img_1 = Image.open(args.img_1_path)
    img_2 = Image.open(args.img_2_path)

    with torch.no_grad():
        img_1 = transform(img_1)
        img_2 = transform(img_2)

        data = torch.stack((img_1, img_2), dim=0).cuda()
        emb_1, emb_2 = model(data).detach().cpu()

        l2_norm = norm(emb_1 - emb_2)
        cos_sim = cosine_similarity(emb_1, emb_2, dim=0)

        print(f"L2 norm: {l2_norm:0.6f}.")
        print(f"Cosine similarity: {cos_sim:0.6f}.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
