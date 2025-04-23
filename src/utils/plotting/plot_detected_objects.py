__all__ = (
    "plot_detected_objects",
)

import typing
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.v2 import functional as F

from common.types import ImageLike, ObjectDetectionTargetsLike

type _ImageOrImageWithTargets = (
    ImageLike | tuple[ImageLike, ObjectDetectionTargetsLike]
)


def plot_detected_objects(
    images: _ImageOrImageWithTargets
        | Sequence[_ImageOrImageWithTargets]
        | Sequence[Sequence[_ImageOrImageWithTargets]],
    row_titles: list[str] | None = None,
    **imshow_kwargs,
):
    if not isinstance(images, Sequence):
        images = [[images]]
    elif len(images) > 0 and not isinstance(images[0], Sequence):
        images = [typing.cast(Sequence[_ImageOrImageWithTargets], images)]

    images = typing.cast(
        Sequence[Sequence[_ImageOrImageWithTargets]],
        images,
    )

    num_rows = len(images)
    num_cols = 0 if num_rows == 0 else len(images[0])

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(images):
        for col_idx, image in enumerate(row):
            boxes = None
            masks = None

            if isinstance(image, tuple):
                image, target = image

                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                else:
                    boxes = target

            image = F.to_image(image)

            if image.dtype.is_floating_point and image.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                image -= image.min()
                image /= image.max()

            image = F.to_dtype(image, torch.uint8, scale=True)

            if boxes is not None:
                image = draw_bounding_boxes(
                    image,
                    boxes,
                    colors="yellow",
                    width=3,
                )

            if masks is not None:
                image = draw_segmentation_masks(
                    image,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(image.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_titles is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_titles[row_idx])

    plt.tight_layout()
