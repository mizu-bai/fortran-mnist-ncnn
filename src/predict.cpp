#include "predict.h"
#include "mnist-12-bin.h"
#include "mat.h"
#include "net.h"
#include "datareader.h"
#include <cstdio>

// mnist-12-opt.param.bin is the BINARY param format. load_param() only accepts
// the text .param format, so it has to go through load_param_bin(); older ncnn
// had a load_param_bin(const unsigned char*) overload but current versions only
// take a DataReader, a FILE* or a path.
//
// The original code called load_param() on the binary blob. That returns 400
// and the return value was discarded, so the failure was silent.
void predict(const float *image, int *label_pred)
{
    // receive input -- ncnn copies out of this Mat, it does not take ownership
    ncnn::Mat in(28, 28, 1, (void *)image);

    // load model
    ncnn::Net net;

    const unsigned char *param_mem = mnist_12_opt_param_bin;
    ncnn::DataReaderFromMemory param_reader(param_mem);
    int r_param = net.load_param_bin(param_reader);
    if (r_param != 0)
    {
        fprintf(stderr, "predict: load_param_bin failed (%d)\n", r_param);
        *label_pred = -1;
        return;
    }

    // load_model() returns the number of bytes consumed, not a status code,
    // so 0 is the failure case -- checking != 0 here would reject success.
    size_t r_model = net.load_model(mnist_12_opt_bin);
    if (r_model == 0)
    {
        fprintf(stderr, "predict: load_model failed\n");
        *label_pred = -1;
        return;
    }

    // extractor
    ncnn::Extractor extractor = net.create_extractor();
    int r_input = extractor.input(0, in);
    if (r_input != 0)
    {
        fprintf(stderr, "predict: extractor.input failed (%d)\n", r_input);
        *label_pred = -1;
        return;
    }

    ncnn::Mat out;
    int r_extract = extractor.extract(8, out);
    if (r_extract != 0)
    {
        fprintf(stderr, "predict: extractor.extract failed (%d)\n", r_extract);
        *label_pred = -1;
        return;
    }

    // The Gemm in mnist-12 is declared 3-in/1-out and should yield a 10-element
    // vector, but ncnn materialises it as w=10 h=8 (80 elements). Reading a
    // flat 10 without checking the shape silently reads the first row only, so
    // guard on it rather than trusting the model.
    if (out.w < 10 || out.empty())
    {
        fprintf(stderr, "predict: unexpected output shape %dx%dx%d\n", out.w, out.h, out.c);
        *label_pred = -1;
        return;
    }
    if (out.w * out.h != 10)
    {
        fprintf(stderr, "predict: warning: output is %dx%dx%d (%d elements), expected 10\n",
                out.w, out.h, out.c, out.w * out.h);
    }

    // get label
    int guessed = -1;
    float guess_exp = -10000000;

    for (int i = 0; i < 10; i++)
    {
        if (out[i] > guess_exp)
        {
            guessed = i;
            guess_exp = out[i];
        }
    }
    *label_pred = guessed;

    extractor.clear();
    net.clear();
}
