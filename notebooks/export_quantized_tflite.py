"""Export fp16 and int8 EfficientSAM prompt models alongside the fp32 pair.

Companion to EfficientSAM_tflite_export.ipynb. That notebook produces the
float32 export; this reuses the same traceable wrappers and re-converts them
under litert-torch quantization recipes, so every precision comes out of one
code path and the graphs stay identical apart from weight dtype.

Run from the repo root:

    /opt/anaconda3/envs/tflite/bin/python notebooks/export_quantized_tflite.py

Writes weights/tflite/efficient_sam_<name>_prompt_<stage>_<suffix>.tflite,
which is the naming PromptSegmenter.Variant expects on the Android side.
"""

import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import litert_torch
from litert_torch.generative.quantize import quant_recipes

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)
from efficient_sam.efficient_sam_encoder import get_abs_pos

MODEL_NAMES = ["vitt", "vits"]
ENCODER_SIZE = 1024
NUM_POINTS = 2
OUTPUT_SIZE = 256
OUT_DIR = "weights/tflite"

# (suffix, encoder recipe, decoder recipe). A None recipe means that stage
# stays float32.
#
# int8 is encoder-only on purpose. Applying full_dynamic_recipe to the decoder
# quantizes the mask upsampler's TRANSPOSE_CONV weights, and Android's kernel
# refuses to allocate the graph:
#
#     transpose_conv.cc:312 weights->type != input->type (INT8 != FLOAT32)
#
# The desktop LiteRT runtime happens to accept it, so this only shows up on
# device. Little is lost: the encoder is where the weights are (95.5 MB vs
# 20.6 MB for vits), and the decoder runs in ~200 ms either way.
RECIPES = [
    ("_fp16", quant_recipes.full_fp16_recipe, quant_recipes.full_fp16_recipe),
    ("_int8", quant_recipes.full_dynamic_recipe, None),
]


class PromptEncoderStage(torch.nn.Module):
    """Stretched 1024x1024 [0,1] image -> cached embedding. Runs once per image."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.image_encoder
        # Fold preprocess() normalization into the graph.
        self.register_buffer("pixel_mean", model.pixel_mean.clone())
        self.register_buffer("pixel_std", model.pixel_std.clone())

        grid = self.encoder.img_size // self.encoder.patch_embed.proj.kernel_size[0]
        with torch.no_grad():
            abs_pos = get_abs_pos(
                self.encoder.pos_embed,
                self.encoder.pretrain_use_cls_token,
                [grid, grid],
            )
        # Constant w.r.t. the input: fold it in instead of recomputing per call.
        self.register_buffer("abs_pos", abs_pos)
        self.grid = grid

    def forward(self, image):
        x = (image - self.pixel_mean) / self.pixel_std
        x = self.encoder.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.abs_pos
        n = self.grid
        x = x.reshape(x.shape[0], n * n, x.shape[3])
        for blk in self.encoder.blocks:
            x = blk(x)
        x = x.reshape(x.shape[0], n, n, x.shape[2])
        return self.encoder.neck(x.permute(0, 3, 1, 2))


class PromptDecoderStage(torch.nn.Module):
    """Cached embedding + one prompt -> 3 candidate masks + IoU scores, UNSORTED.

    No in-graph sort: torch.argsort lowers to STABLEHLO_SORT, which the LiteRT
    runtime has no kernel for. Host code takes the argmax instead.
    """

    def __init__(self, model, output_size):
        super().__init__()
        self.model = model
        self.output_size = output_size
        self.max_pts = model.decoder_max_num_input_points

    def forward(self, image_embeddings, point_coords, point_labels):
        batch_size, num_queries, num_pts, _ = point_coords.shape

        # Same pad/truncate convention as EfficientSam.predict_masks.
        if num_pts > self.max_pts:
            point_coords = point_coords[:, :, : self.max_pts, :]
            point_labels = point_labels[:, :, : self.max_pts]
        elif num_pts < self.max_pts:
            point_coords = F.pad(
                point_coords, (0, 0, 0, self.max_pts - num_pts), value=-1.0
            )
            point_labels = F.pad(point_labels, (0, self.max_pts - num_pts), value=-1.0)

        sparse_embeddings = self.model.prompt_encoder(
            point_coords.reshape(batch_size * num_queries, self.max_pts, 2),
            point_labels.reshape(batch_size * num_queries, self.max_pts),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings,
            self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=True,
        )
        num_predictions = low_res_masks.shape[1]

        masks = F.interpolate(
            low_res_masks,
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.reshape(
            batch_size, num_queries, num_predictions, self.output_size, self.output_size
        )
        iou_predictions = iou_predictions.reshape(
            batch_size, num_queries, num_predictions
        )
        return masks, iou_predictions


def build_wrappers():
    # The VIT-small checkpoint ships zipped because it is >100MB.
    if not os.path.exists("weights/efficient_sam_vits.pt"):
        with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip") as z:
            z.extractall("weights")

    builders = {"vitt": build_efficient_sam_vitt, "vits": build_efficient_sam_vits}
    out = {}
    for name in MODEL_NAMES:
        m = builders[name]().eval()
        out[name] = (
            PromptEncoderStage(m).eval(),
            PromptDecoderStage(m, OUTPUT_SIZE).eval(),
        )
    return out


def paths_for(name, suffix):
    return (
        f"{OUT_DIR}/efficient_sam_{name}_prompt_encoder{suffix}.tflite",
        f"{OUT_DIR}/efficient_sam_{name}_prompt_decoder{suffix}.tflite",
    )


def convert_pair(name, wrappers, suffix, enc_recipe, dec_recipe):
    enc_w, dec_w = wrappers[name]
    enc_path, dec_path = paths_for(name, suffix)

    enc_sample = (torch.rand(1, 3, ENCODER_SIZE, ENCODER_SIZE),)
    with torch.no_grad():
        embed_shape = tuple(enc_w(*enc_sample).shape)

    t = time.time()
    litert_torch.convert(
        enc_w, enc_sample, quant_config=enc_recipe() if enc_recipe else None
    ).export(enc_path)
    enc_mb = os.path.getsize(enc_path) / 1e6
    print(f"  encoder {embed_shape} {enc_mb:6.1f} MB in {time.time()-t:5.1f}s")

    dec_sample = (
        torch.rand(*embed_shape),
        torch.rand(1, 1, NUM_POINTS, 2) * ENCODER_SIZE,
        torch.ones(1, 1, NUM_POINTS),
    )
    t = time.time()
    litert_torch.convert(
        dec_w, dec_sample, quant_config=dec_recipe() if dec_recipe else None
    ).export(dec_path)
    dec_mb = os.path.getsize(dec_path) / 1e6
    note = "" if dec_recipe else "  (float32: see RECIPES)"
    print(f"  decoder {dec_mb:6.1f} MB in {time.time()-t:5.1f}s{note}")
    return enc_path, dec_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wrappers = build_wrappers()
    for name in MODEL_NAMES:
        for suffix, enc_recipe, dec_recipe in RECIPES:
            print(f"\n{name}{suffix}:")
            try:
                convert_pair(name, wrappers, suffix, enc_recipe, dec_recipe)
            except Exception as e:
                # One failing recipe should not lose the others; report and move on.
                print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
