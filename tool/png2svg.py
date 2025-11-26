#!/usr/bin/env python3

from PIL import Image
import numpy as np
import base64
import io
import os


def remove_near_white_background(
    input_path: str,
    output_png_path: str,
    white_threshold: int = 240,
    std_threshold: int = 15,
    extra_margin: int = 0,
) -> None:
    """
    - Removes near-white background by setting alpha=0.
    - Crops to the tight bounding box of non-transparent pixels.
    - Saves a cleaned PNG with transparency.

    Args:
        input_path: path to input PNG.
        output_png_path: path to cleaned PNG to save.
        white_threshold: pixels with all channels > this are considered white-ish.
        std_threshold: how similar R,G,B must be to be considered 'white-ish'.
        extra_margin: extra pixels of border to keep around content.
    """
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)

    # Separate channels
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    # Heuristic: near-white if all channels high AND not very colorful
    mean_rgb = (r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)) / 3.0
    std_rgb = np.std(arr[..., :3].astype(np.int16), axis=-1)

    near_white = (mean_rgb >= white_threshold) & (std_rgb <= std_threshold)

    # Make near-white fully transparent
    a[near_white] = 0
    arr[..., 3] = a

    # Find bounding box of non-transparent content
    alpha_mask = a > 0
    if not np.any(alpha_mask):
        # If everything became transparent, just save the original alpha-ed image
        cleaned = Image.fromarray(arr, mode="RGBA")
        cleaned.save(output_png_path)
        print("Warning: all pixels detected as background; saved un-cropped PNG.")
        return

    ys, xs = np.where(alpha_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Apply optional margin
    h, w = a.shape
    y_min = max(y_min - extra_margin, 0)
    x_min = max(x_min - extra_margin, 0)
    y_max = min(y_max + extra_margin, h - 1)
    x_max = min(x_max + extra_margin, w - 1)

    cropped_arr = arr[y_min : y_max + 1, x_min : x_max + 1, :]
    cleaned = Image.fromarray(cropped_arr, mode="RGBA")
    cleaned.save(output_png_path)
    print(f"Saved cleaned PNG to {output_png_path}")


def embed_png_in_svg(png_path: str, svg_path: str) -> None:
    """
    Embeds a PNG image inside an SVG as a base64-encoded <image>.

    Args:
        png_path: path to PNG.
        svg_path: path to SVG to create.
    """
    img = Image.open(png_path)
    width, height = img.size

    # Encode PNG to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("ascii")

    svg_template = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="{width}"
    height="{height}"
    viewBox="0 0 {width} {height}">
  <image
    x="0"
    y="0"
    width="{width}"
    height="{height}"
    xlink:href="data:image/png;base64,{b64_data}" />
</svg>
"""

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_template)

    print(f"Saved SVG to {svg_path}")


if __name__ == "__main__":
    # Example usage:
    input_png = "static/images/anchordream.png"
    cleaned_png = "static/images/anchordream_cleaned.png"
    output_svg = "static/images/anchordream.svg"

    # 1) Remove near-white background and crop
    remove_near_white_background(
        input_path=input_png,
        output_png_path=cleaned_png,
        white_threshold=245,  # adjust if your background is darker/lighter
        std_threshold=15,     # adjust if your background is slightly colored
        extra_margin=4,       # small border; set 0 for strict minimal border
    )

    # 2) Convert cleaned PNG to SVG (embedded raster)
    embed_png_in_svg(cleaned_png, output_svg)
