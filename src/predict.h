#ifndef PREDICT_HEADER_
#define PREDICT_HEADER_

extern "C" {
    // image points at 28*28 floats in row-major order (ncnn layout, not
    // Fortran's column-major). label_pred receives the argmax class, or -1
    // when inference fails.
    void predict(const float *image, int *label_pred);
}

#endif
