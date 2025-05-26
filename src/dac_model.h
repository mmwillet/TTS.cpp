#ifndef dac_model_h
#define dac_model_h

#include "tts_model.h"
#include <map>

enum dac_tensor {
  DAC_ENCODER_IN_KERNEL,
  DAC_ENCODER_IN_BIAS,
  DAC_ENCODER_OUT_KERNEL,
  DAC_ENCODER_OUT_BIAS,
  DAC_ENCODER_SNAKE_ALPHA,
  DAC_ENCODER_LAYER_SNAKE_ALPHA,
  DAC_ENCODER_LAYER_OUT_KERNEL,
  DAC_ENCODER_LAYER_OUT_BIAS,
  DAC_ENCODER_LAYER_RES_BLK_IN_SNAKE,
  DAC_ENCODER_LAYER_RES_BLK_OUT_SNAKE,
  DAC_ENCODER_LAYER_RES_BLK_IN_KERNEL,
  DAC_ENCODER_LAYER_RES_BLK_OUT_KERNEL,
  DAC_ENCODER_LAYER_RES_BLK_IN_BIAS,
  DAC_ENCODER_LAYER_RES_BLK_OUT_BIAS,
  DAC_QUANTIZER_LAYER_IN_KERNEL,
  DAC_QUANTIZER_LAYER_IN_BIAS,
  DAC_QUANTIZER_LAYER_OUT_KERNEL,
  DAC_QUANTIZER_LAYER_OUT_BIAS,
  DAC_QUANTIZER_LAYER_CODEBOOK
};

struct dac_residual_unit {
  struct ggml_tensor *in_snake_alpha;
  struct ggml_tensor *in_conv_kernel;
  struct ggml_tensor *in_conv_bias;
  struct ggml_tensor *out_snake_alpha;
  struct ggml_tensor *out_conv_kernel;
  struct ggml_tensor *out_conv_bias;
};

struct dac_layer {
  struct ggml_tensor *snake_alpha_in;
  struct ggml_tensor *out_conv_kernel;
  struct ggml_tensor *out_conv_bias;

  uint32_t padding;
  uint32_t stride;

  std::vector<dac_residual_unit> residual_blocks;
};

struct dac_quantize_layer {
  struct ggml_tensor *out_proj_kernel;
  struct ggml_tensor *out_proj_bias;
  struct ggml_tensor *codebook;
};

// this struct maintains the static tensors for the dac audio decoder graph.
// As such, this is designed to contain basic configuration and ggml tensor
// support for DAC. The dac_runner describes how the graph is built and run.
struct dac_model : tts_model {
  // These configs  are essentially built for the 44khZ 8kbps standard DAC model
  // audio encoder and decoder
  uint32_t n_layers = 4;
  uint32_t n_heads = 9;
  uint32_t up_sampling_factor = 512;
  uint32_t max_generation_size = 2580;

  struct ggml_tensor *in_conv_kernel;
  struct ggml_tensor *in_conv_bias;
  struct ggml_tensor *out_conv_kernel;
  struct ggml_tensor *out_conv_bias;
  struct ggml_tensor *snake_alpha;
  std::vector<dac_layer> layers;
  std::vector<dac_quantize_layer> quantizer_layers;

  void assign_weight(std::string name, ggml_tensor *weight);
  void prep_constants(gguf_context *meta);
  void prep_layers(gguf_context *meta);
  void setup_from_file(gguf_context *meta_ctx, ggml_context *load_context,
                       bool cpu_only) {
    prep_layers(meta_ctx);
    prep_constants(meta_ctx);
    tts_model::setup_from_file(meta_ctx, load_context, cpu_only,
                               "audio_encoder");
  }
};

// for loading DAC model from gguf file
void assign_residual_unit(dac_model *model, dac_residual_unit *layer,
                          std::string name, ggml_tensor *tensor);
void assign_dac_layer(dac_model *model, dac_layer *layer, std::string name,
                      ggml_tensor *tensor);
void assign_quantizer_layer(dac_model *model, dac_quantize_layer layer,
                            std::string name, ggml_tensor *tensor);
void assign_to_audio_encoder(dac_model *model, std::string name,
                             ggml_tensor *tensor);

// the context used for running the dac model
struct dac_context : runner_context {
  dac_context(dac_model *model, int n_threads)
      : runner_context(n_threads), model(model){};

  struct dac_model *model;

  size_t logits_size = 0; // capacity (of floats) for logits
  float *logits = nullptr;

  struct ggml_tensor *inp_tokens;

  void build_schedule() { runner_context::build_schedule(model->max_nodes()); }
};

struct dac_context *build_new_dac_context(struct dac_model *model,
                                          int n_threads, bool use_cpu = true);

struct dac_ubatch {
  uint32_t *input_tokens;
  uint32_t sequence_length;
};

static struct ggml_tensor *
dac_build_audio_inputs(struct ggml_context *ctx, struct dac_context *dctx,
                       const dac_ubatch &batch,
                       std::vector<dac_quantize_layer> layers);
static struct ggml_tensor *build_residual_unit(ggml_context *ctx,
                                               struct ggml_tensor *cur,
                                               dac_residual_unit &u,
                                               int padding, int dilation);
static struct ggml_tensor *build_decoder_block(ggml_context *ctx,
                                               struct ggml_tensor *cur,
                                               dac_layer &l,
                                               struct dac_context *dctx);

// This struct is intended to manage the dac model's graph compilation and
// compute function.
struct dac_runner : tts_runner {
  dac_runner(dac_model *model, dac_context *context)
      : model(model), dctx(context){};
  ~dac_runner() {
    if (ctx) {
      ggml_free(ctx);
    }
    model->free();
    delete model;
    delete dctx;
  }
  dac_model *model;
  dac_context *dctx;

  void init_build() { tts_runner::init_build(&dctx->buf_compute_meta); }

  void prepare_post_load();
  struct ggml_cgraph *build_dac_graph(dac_ubatch &batch);
  void run(uint32_t *input_tokens, uint32_t sequence_length,
           struct tts_response *outputs);
};

#endif
