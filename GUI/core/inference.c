#include "inference.h"

#if defined(_WIN32)
#include <windows.h>
#endif

// Global pointers for the ORT environment and session
const OrtApi* g_ort = NULL;
OrtEnv* env = NULL;
OrtSession* session = NULL;
OrtMemoryInfo* memory_info = NULL;

static int log_and_release_status(const char* context, OrtStatus* status) {
    if (status == NULL) {
        return HYBRID_OK;
    }
    printf("%s: %s\n", context, g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
    return HYBRID_ERR_ORT_API;
}

#if defined(_WIN32)
static wchar_t* utf8_to_wide(const char* utf8) {
    int len_wide;
    wchar_t* wide;

    if (utf8 == NULL) {
        return NULL;
    }

    len_wide = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (len_wide <= 0) {
        return NULL;
    }

    wide = (wchar_t*)malloc((size_t)len_wide * sizeof(wchar_t));
    if (wide == NULL) {
        return NULL;
    }

    if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide, len_wide) <= 0) {
        free(wide);
        return NULL;
    }

    return wide;
}
#endif

HYBRID_API
int init_engine(const char* model_path) {
    OrtStatus* status = NULL;
#if defined(_WIN32)
    wchar_t* model_path_wide = NULL;
#endif

    if (model_path == NULL) {
        return HYBRID_ERR_INVALID_ARGUMENT;
    }

    if (session != NULL) {
        cleanup_engine();
    }

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        return HYBRID_ERR_ORT_API;
    }

    status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HybridEngine", &env);
    if (status != NULL) {
        return log_and_release_status("CreateEnv failed", status);
    }
    
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        cleanup_engine();
        return log_and_release_status("CreateSessionOptions failed", status);
    }

    status = g_ort->SetIntraOpNumThreads(session_options, 1);
    if (status != NULL) {
        g_ort->ReleaseSessionOptions(session_options);
        cleanup_engine();
        return log_and_release_status("SetIntraOpNumThreads failed", status);
    }

    status = g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
    if (status != NULL) {
        g_ort->ReleaseSessionOptions(session_options);
        cleanup_engine();
        return log_and_release_status("SetSessionGraphOptimizationLevel failed", status);
    }

    // Load the model. ONNX Runtime expects a wide path on Windows.
#if defined(_WIN32)
    model_path_wide = utf8_to_wide(model_path);
    if (model_path_wide == NULL) {
        g_ort->ReleaseSessionOptions(session_options);
        cleanup_engine();
        return HYBRID_ERR_INVALID_ARGUMENT;
    }

    status = g_ort->CreateSession(env, model_path_wide, session_options, &session);
#else
    status = g_ort->CreateSession(env, model_path, session_options, &session);
#endif
    g_ort->ReleaseSessionOptions(session_options);
#if defined(_WIN32)
    free(model_path_wide);
#endif
    
    if (status != NULL) {
        cleanup_engine();
        log_and_release_status("Failed to load model", status);
        return HYBRID_ERR_MODEL_LOAD;
    }

    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        cleanup_engine();
        return log_and_release_status("CreateCpuMemoryInfo failed", status);
    }

    return HYBRID_OK;
}


HYBRID_API
int run_hybrid_inference(
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
    int64_t output_dim)
{
    if (!session || !memory_info) {
        return HYBRID_ERR_NOT_INITIALIZED;
    }

    if (node_features == NULL || edge_index == NULL || edge_attr == NULL || batch_index == NULL ||
        input_ids == NULL || attention_mask == NULL || output_properties == NULL) {
        return HYBRID_ERR_INVALID_ARGUMENT;
    }

    if (num_nodes <= 0 || node_feat_dim <= 0 || num_edges <= 0 || edge_feat_dim <= 0 || seq_len <= 0 || output_dim <= 0) {
        return HYBRID_ERR_INVALID_ARGUMENT;
    }

    OrtStatus* status = NULL;

    // Hybrid model inputs:
    // x [num_nodes, node_feat_dim], edge_index [2, num_edges], edge_attr [num_edges, edge_feat_dim],
    // batch [num_nodes], input_ids [1, seq_len], attention_mask [1, seq_len]
    int64_t x_shape[] = {num_nodes, node_feat_dim};
    int64_t edge_index_shape[] = {2, num_edges};
    int64_t edge_attr_shape[] = {num_edges, edge_feat_dim};
    int64_t batch_shape[] = {num_nodes};
    int64_t token_shape[] = {1, seq_len};

    OrtValue* input_tensors[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)node_features,
        (size_t)(num_nodes * node_feat_dim * (int64_t)sizeof(float)),
        x_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensors[0]);
    if (status != NULL) {
        log_and_release_status("Create tensor for x failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)edge_index,
        (size_t)(2 * num_edges * (int64_t)sizeof(int64_t)),
        edge_index_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensors[1]);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensors[0]);
        log_and_release_status("Create tensor for edge_index failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)edge_attr,
        (size_t)(num_edges * edge_feat_dim * (int64_t)sizeof(float)),
        edge_attr_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensors[2]);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensors[0]);
        g_ort->ReleaseValue(input_tensors[1]);
        log_and_release_status("Create tensor for edge_attr failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)batch_index,
        (size_t)(num_nodes * (int64_t)sizeof(int64_t)),
        batch_shape,
        1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensors[3]);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensors[0]);
        g_ort->ReleaseValue(input_tensors[1]);
        g_ort->ReleaseValue(input_tensors[2]);
        log_and_release_status("Create tensor for batch failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)input_ids,
        (size_t)(seq_len * (int64_t)sizeof(int64_t)),
        token_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensors[4]);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensors[0]);
        g_ort->ReleaseValue(input_tensors[1]);
        g_ort->ReleaseValue(input_tensors[2]);
        g_ort->ReleaseValue(input_tensors[3]);
        log_and_release_status("Create tensor for input_ids failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)attention_mask,
        (size_t)(seq_len * (int64_t)sizeof(int64_t)),
        token_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensors[5]);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensors[0]);
        g_ort->ReleaseValue(input_tensors[1]);
        g_ort->ReleaseValue(input_tensors[2]);
        g_ort->ReleaseValue(input_tensors[3]);
        g_ort->ReleaseValue(input_tensors[4]);
        log_and_release_status("Create tensor for attention_mask failed", status);
        return HYBRID_ERR_TENSOR_CREATE;
    }

    // Input/output names must match ONNX export names.
    const char* input_names[] = {
        "x",
        "edge_index",
        "edge_attr",
        "batch",
        "input_ids",
        "attention_mask"
    };
    const char* output_names[] = {"predicted_properties"};

    OrtValue* output_tensor = NULL;
    status = g_ort->Run(
        session,
        NULL,
        input_names,
        (const OrtValue* const*)input_tensors,
        6,
        output_names,
        1,
        &output_tensor);

    if (status != NULL) {
        int i;
        for (i = 0; i < 6; i++) {
            g_ort->ReleaseValue(input_tensors[i]);
        }
        log_and_release_status("Inference Run failed", status);
        return HYBRID_ERR_RUN;
    }

    float* out_arr = NULL;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&out_arr);
    if (status != NULL || out_arr == NULL) {
        int i;
        for (i = 0; i < 6; i++) {
            g_ort->ReleaseValue(input_tensors[i]);
        }
        g_ort->ReleaseValue(output_tensor);
        if (status != NULL) {
            log_and_release_status("GetTensorMutableData failed", status);
        }
        return HYBRID_ERR_OUTPUT;
    }

    for (int64_t i = 0; i < output_dim; i++) {
        output_properties[i] = out_arr[i];
    }

    g_ort->ReleaseValue(output_tensor);
    for (int i = 0; i < 6; i++) {
        g_ort->ReleaseValue(input_tensors[i]);
    }

    return HYBRID_OK;
}

HYBRID_API
int run_inference(
    float* node_features, int num_nodes, int node_feat_dim,
    int64_t* edge_indices, int num_edges,
    const char* smiles,
    float* output_properties, int output_dim) 
{
    (void)node_features;
    (void)num_nodes;
    (void)node_feat_dim;
    (void)edge_indices;
    (void)num_edges;
    (void)smiles;
    (void)output_properties;
    (void)output_dim;
    return HYBRID_ERR_RUN;
}

// 3. Cleanup
HYBRID_API
void cleanup_engine() {
    if (!g_ort) {
        memory_info = NULL;
        session = NULL;
        env = NULL;
        return;
    }

    if (memory_info) {
        g_ort->ReleaseMemoryInfo(memory_info);
        memory_info = NULL;
    }
    if (session) {
        g_ort->ReleaseSession(session);
        session = NULL;
    }
    if (env) {
        g_ort->ReleaseEnv(env);
        env = NULL;
    }
}