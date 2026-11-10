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

## A note on ncnn's ARM Gemm and packing

The model's final `Gemm` produced a `10x8` result instead of `10x1` on ARM,
which made every digit predict the same class. The cause is in ncnn, not here:

- On ARM, `create_layer_cpu` returns a `Gemm_arm` (a `Gemm` subclass), so the
  plain `Gemm::forward` — which computes `M = A.h = 1` correctly — is never the
  one that runs.
- `Gemm_arm::forward` computes `M = A.h * A.elempack`. The `A` feeding this
  Gemm is a packed Mat (`w=64 h=1 elempack=4` in fp32, and `elempack=8` under
  fp16), so M becomes 4 or 8 instead of 1 and the result Mat is created as
  `10x4` or `10x8`.
- The labels were then read from `out[0..9]`, i.e. the first row of a
  multi-row matrix.

`predict.cpp` therefore turns `use_packing_layout` and the fp16 storage paths
off for this net. That keeps A unpacked, M stays 1, and the output is `10x1`
as the graph says. The trade-off is some speed: packing exists so the NEON
kernels can process several values at once. For a 28x28 MNIST net that is
irrelevant; for a larger model you would want to keep packing on and instead
avoid a packed Mat reaching a Gemm.

With that in place the bundled images classify correctly: `2.dat` → 2,
`4.dat` → 4, `7.dat` → 7.

## References

- [ncnn](https://github.com/Tencent/ncnn)
- [ncnn_on_xr806](https://github.com/nihui/ncnn_on_xr806)
- [MNIST - Handwritten Digit Recognition](https://github.com/onnx/models/tree/main/vision/classification/mnist)
