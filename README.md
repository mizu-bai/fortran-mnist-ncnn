# fortran-mnist-ncnn

Call ncnn from Fortran via mix compiling.

## Introduction

How this demo works?

1. Image

    As Fortran is mainly used in scientific computation, there seems to be no workable OpenCV binding for Fortran or other computer vision libraries. The repo [ncnn_on_xr806
](https://github.com/nihui/ncnn_on_xr806) inspired me for the image was stored in a header file in the format of `const unsigned char` array and `ncnn::Mat::from_pixels` is ready for it. In this repo the image is stored in a ASCII-text file with 28 rows and 28 columns, it should be in `csv` or `tsv` like format. Here are some so-called images in `image/` directory.

2. Main Program

    The main program `main.f90` is written in Fortran, it parses command line arguments, loads image, call ncnn binding, and output the most possible label of input image.

3. ncnn Binding

    ncnn Binding includes 4 files:

    - `ncnn_mnist.f90`
    - `predict.h`
    - `predict.cpp`
    - `mnist-12-bin.h`

    `ncnn_mnist.f90` is the wrapper of function `predict`, it interacts with `predict.cpp` via `extern "C"` in `predict.h`. `mnist-12-bin.h` is the model waits ncnn to load. In `predict.cpp`, function `predict` initializes an `ncnn::Mat` from a 2d Fortran array, loads the model, and predicts the label. Finally, the predicted label is passed to main program.

    The image array is stored column-major by Fortran but read row-wise from the file, which makes the underlying memory row-major — the layout `ncnn::Mat` expects. See the comment in `main.f90`.

## Build

```sh
mkdir build && cd build
cmake .. -Dncnn_DIR=/path/to/ncnn
make
cd ..
./build/mnist image/4.dat
```

`ncnn_DIR` accepts either an install prefix or a build tree containing
`src/libncnn.a` and the generated `src/platform.h`. `NCNN_DIR` as an
environment variable works too, as does naming the archive directly with
`-Dncnn_LIBRARY=/path/to/libncnn.a`.

gfortran is used by default; override with `-DCMAKE_Fortran_COMPILER=...`.
GNU and Intel compilers have been tested.

## Known issue: wrong predictions

`predict` currently reports the wrong label for the bundled images. The failure
is **not** in this repo's Fortran/C++ code — it is reproducible through ncnn's
own C API and with a freshly regenerated optimised model, so it is a property
of the `mnist-12` model as ncnn executes it.

What is established:

- The image is read correctly; `image/4.dat` renders as a 4.
- The model loads: 9 layers, weights load, and every blob up to the last has
  the expected shape (`28x28x1`, `28x28x8`, `14x14x8`, `14x14x16`, `4x4x16`,
  `256x1`).
- The final `Gemm` materialises as `w=10 h=8 c=1` — 80 elements — where the
  ONNX graph says it should be 10. `Gemm::forward` computes
  `top_blob.create(N, M)` with `M = A.h = 1` and `N = 10`, so `h` should be 1.
- The original code read `out[0..9]` unconditionally, i.e. the first row of an
  8-row matrix, and discarded every ncnn return code.

`predict.cpp` now checks each return code and warns when the output shape is
not the expected 10 elements, so the failure is visible instead of silent.
Fixing it properly means sorting out why ncnn's Gemm produces an 8-row output
for this model — re-optimising the model, adjusting the Gemm parameters, or
converting from the ONNX again — which is a model-side problem rather than a
binding one.

## References

- [ncnn](https://github.com/Tencent/ncnn)
- [ncnn_on_xr806](https://github.com/nihui/ncnn_on_xr806)
- [MNIST - Handwritten Digit Recognition](https://github.com/onnx/models/tree/main/vision/classification/mnist)
