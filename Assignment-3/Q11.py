"""
SMAS-1 Assignment 3 — Q11: Interactive Image Transformation Toolbox
Run locally: python q11_interactive_toolbox.py

Requires: numpy, matplotlib, scipy, pillow
    pip install numpy matplotlib scipy pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from PIL import Image


def load_image(path=None):
    """Load an image from disk as an (H,W,3) float array in [0,1]."""
    if path is None:
        path = input("Enter path to your image file (jpg/png): ").strip()
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float64) / 255.0


def show(img, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.show()


def apply_centered_linear_map(image, A, center):
    """
    Apply y = A(x - c) + c to every pixel using INVERSE mapping so the
    output image has no holes.
    """
    Ainv = np.linalg.pinv(A)
    offset = center - Ainv @ center
    swap = np.array([[0, 1], [1, 0]])
    M_rc = swap @ Ainv @ swap
    offset_rc = swap @ offset

    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[:, :, c] = affine_transform(
            image[:, :, c], M_rc, offset=offset_rc,
            order=1, mode="constant", cval=0.0
        )
    return out


def get_transform_matrix():
    """Prompt the user for an operation and return the 2x2 matrix (x,y) plus a label."""
    print("\nOptions: Rotate | Resize | Flip | Shear | Custom Matrix | Reset")
    op = input("Choose an operation: ").strip().lower()

    if op == "rotate":
        theta = float(input("Enter rotation angle in degrees: "))
        t = np.deg2rad(theta)
        A = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t),  np.cos(t)]])
        return A, f"Rotate {theta} deg"

    elif op == "resize":
        sx = float(input("Enter x scale factor: "))
        sy = float(input("Enter y scale factor: "))
        A = np.array([[sx, 0], [0, sy]])
        return A, f"Resize sx={sx}, sy={sy}"

    elif op == "flip":
        axis = input("Flip about which axis? (x/y): ").strip().lower()
        if axis == "x":
            A = np.array([[1, 0], [0, -1]])
        else:
            A = np.array([[-1, 0], [0, 1]])
        return A, f"Flip about {axis}-axis"

    elif op == "shear":
        kind = input("Shear direction? (horizontal/vertical): ").strip().lower()
        k = float(input("Enter shear factor: "))
        if kind.startswith("h"):
            A = np.array([[1, k], [0, 1]])
        else:
            A = np.array([[1, 0], [k, 1]])
        return A, f"Shear {kind} k={k}"

    elif op in ("custom matrix", "custom"):
        print("Enter the 2x2 matrix row by row (space-separated), e.g. '1 0'")
        r1 = [float(x) for x in input("Row 1: ").split()]
        r2 = [float(x) for x in input("Row 2: ").split()]
        A = np.array([r1, r2])
        return A, "Custom matrix"

    elif op == "reset":
        return None, "Reset"

    else:
        print("Unrecognised option, try again.")
        return get_transform_matrix()


def run_toolbox():
    print("=== Interactive Image Transformation Toolbox ===")
    original = load_image()
    current = original.copy()
    H, W, _ = original.shape
    center = np.array([W / 2.0, H / 2.0])
    show(current, "Current image")

    while True:
        A, label = get_transform_matrix()
        if label == "Reset":
            current = original.copy()
            show(current, "Reset to original")
        else:
            current = apply_centered_linear_map(current, A, center)
            show(current, label)

        again = input("Apply another transform? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break


if __name__ == "__main__":
    run_toolbox()
