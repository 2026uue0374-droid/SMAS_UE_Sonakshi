# SMAS-1 Assignment 3
## Image Transformation using Python

This assignment explores image transformations using concepts from linear algebra and Python.

### Q10 – Transform an Actual Image

An actual image was uploaded into a Python notebook and treated as a collection of 2-D pixel coordinate vectors about the image centre.

The following transformation matrices were applied:

- A1 – Scaling
- A2 – 90° Rotation
- A3 – Horizontal Shear
- A4 – Reflection in the y-axis
- A5 – Projection onto the x-axis

For each transformation, the transformed basis vectors, transformed image, rank, information loss, and the relationship between the visual effect and the columns of the matrix were analyzed.

### Q11 – Interactive Image Transformation Toolbox

An interactive image transformation toolbox was developed in Python/Google Colab.

The toolbox allows the user to upload a JPG/PNG image and select different transformations through a menu.

Available operations:

- Rotate
- Resize
- Flip
- Shear
- Custom Matrix
- Reset

The required parameters for each operation can be entered by the user, and the transformed image is displayed.

### Files

- `Python_Toolbox/` – Python/Colab notebook containing the implementation.
- `sample_photo.jpg` – Sample image used for testing.
- `README.md` – Description of the assignment and files.
