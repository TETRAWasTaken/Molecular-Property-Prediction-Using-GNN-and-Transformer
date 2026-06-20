#ifndef HYBRID_GUI_INFERENCE_H
#define HYBRID_GUI_INFERENCE_H

#include "onnxruntime_c_api.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(_MSC_VER)
#define HYBRID_API __declspec(dllexport)
#else
#define HYBRID_API __attribute__((dllexport))
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HYBRID_API __attribute__((visibility("default")))
#else
#define HYBRID_API
#endif

enum {
	HYBRID_OK = 0,
	HYBRID_ERR_NOT_INITIALIZED = -1,
	HYBRID_ERR_ORT_API = -2,
	HYBRID_ERR_MODEL_LOAD = -3,
	HYBRID_ERR_INVALID_ARGUMENT = -4,
	HYBRID_ERR_TENSOR_CREATE = -5,
	HYBRID_ERR_RUN = -6,
	HYBRID_ERR_OUTPUT = -7
};

/*
 * Initializes ONNX Runtime and loads an ONNX model session.
 */
HYBRID_API int init_engine(const char* model_path);

/*
 * Hybrid model inference.
 *
 * Expected model inputs (export names):
 * - x:              float32 [num_nodes, node_feat_dim]
 * - edge_index:     int64   [2, num_edges]
 * - edge_attr:      float32 [num_edges, edge_feat_dim]
 * - batch:          int64   [num_nodes]
 * - input_ids:      int64   [1, seq_len]
 * - attention_mask: int64   [1, seq_len]
 *
 * Expected output:
 * - predicted_properties: float32 [1, output_dim] (or [output_dim])
 */
HYBRID_API int run_hybrid_inference(
	const float* node_features,
	int64_t num_nodes,
	int64_t node_feat_dim,
	const int64_t* edge_index,
	int64_t num_edges,
	const float* edge_attr,
	int64_t edge_feat_dim,
	const int64_t* batch_index,
	const int64_t* input_ids,
	int64_t seq_len,
	const int64_t* attention_mask,
	float* output_properties,
	int64_t output_dim
);

/*
 * Backward-compatible graph-only entry point.
 * For hybrid ONNX models this will return HYBRID_ERR_RUN if required text inputs are absent.
 */
HYBRID_API int run_inference(
	float* node_features,
	int num_nodes,
	int node_feat_dim,
	int64_t* edge_indices,
	int num_edges,
	const char* smiles,
	float* output_properties,
	int output_dim
);

/*
 * Releases all engine resources.
 */
HYBRID_API void cleanup_engine(void);

#ifdef __cplusplus
}
#endif

#endif