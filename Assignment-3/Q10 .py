"""
SMAS-1 Assignment 3 — Q10: Transform an actual image
Run locally: python q10_transform_image.py

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
    output image has no holes. scipy.ndimage.affine_transform maps an
    output coordinate o -> input coordinate via: input = M @ o + offset.
    We want: input = A^{-1} @ (o - c) + c  =>  M = A^{-1}, offset = c - M@c.
    If A is singular, a pseudo-inverse is used purely for resampling.
    """
    Ainv = np.linalg.pinv(A)
    offset = center - Ainv @ center
    # affine_transform works in (row, col) = (y, x) order, our A is in (x, y),
    # so build the equivalent (row,col) matrix/offset by swapping axes.
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


def describe_effect(name):
    sentences = {
        "A1 (scaling)":
            "The first column [2,0] stretches e1 (x) to double length and the "
            "second column [0,0.5] shrinks e2 (y) to half length, so the image "
            "appears stretched horizontally and squeezed vertically.",
        "A2 (90 deg rotation)":
            "Column 1 [0,1] shows e1 mapped onto the +y direction and column 2 "
            "[-1,0] shows e2 mapped onto the -x direction, matching the visible "
            "90 degree counter-clockwise rotation of the image.",
        "A3 (horizontal shear)":
            "Column 1 [1,0] leaves e1 fixed while column 2 [1,1] tilts e2 to the "
            "right, which is exactly the horizontal slanting/shear seen in the image.",
        "A4 (reflection y-axis)":
            "Column 1 [-1,0] flips e1 to point in -x while column 2 [0,1] keeps "
            "e2 fixed, producing the left-right mirror flip visible in the image.",
        "A5 (projection onto x)":
            "Column 1 [1,0] keeps e1 but column 2 [0,0] collapses e2 to the zero "
            "vector, so every pixel's height information is destroyed and the "
            "image is flattened onto a single horizontal line.",
    }
    return sentences[name]


def main():
    img = load_image()
    show(img, "Original image")

    H, W, _ = img.shape
    center = np.array([W / 2.0, H / 2.0])  # (x, y) image centre

    matrices = {
        "A1 (scaling)":           np.array([[2, 0], [0, 0.5]], dtype=float),
        "A2 (90 deg rotation)":   np.array([[0, -1], [1, 0]],  dtype=float),
        "A3 (horizontal shear)":  np.array([[1, 1], [0, 1]],   dtype=float),
        "A4 (reflection y-axis)": np.array([[-1, 0], [0, 1]],  dtype=float),
        "A5 (projection onto x)": np.array([[1, 0], [0, 0]],   dtype=float),
    }

    for name, A in matrices.items():
        print("=" * 70)
        print(name)
        print("Matrix A =\n", A)

        # (a) T(e1) and T(e2) are simply the columns of A
        e1, e2 = np.array([1, 0]), np.array([0, 1])
        print(f"(a) T(e1) = A @ e1 = {A @ e1}")
        print(f"    T(e2) = A @ e2 = {A @ e2}")

        # (b) display transformed image
        A_vis = A.copy()
        note = ""
        if name.startswith("A5"):
            A_vis = np.array([[1, 0], [0, 1e-6]])  # tiny nonzero y-scale, display only
            note = "  (visualised with y-scale=1e-6; TRUE matrix is [[1,0],[0,0]])"
        transformed = apply_centered_linear_map(img, A_vis, center)
        show(transformed, name + note)

        # (c) rank
        rank = np.linalg.matrix_rank(A)
        print(f"(c) rank(A) = {rank}")

        # (d) dimension/information loss
        if rank < 2:
            print("(d) YES — rank < 2, so the transform collapses the plane onto a "
                  "line: one full dimension of information (all y-variation) is lost "
                  "and cannot be recovered.")
        else:
            print("(d) NO — rank = 2, A is invertible, so no dimension is lost "
                  "(the transform is reversible), even though the image looks visually "
                  "different.")

        # (e) one-sentence link between visual effect and columns of A
        print("(e)", describe_effect(name))


if __name__ == "__main__":
    main()
