/* Torch-free C ABI over ktt::bridge::* (tt_bridge.hpp). Handles are opaque `ktt_h`,
 * freed with ktt_release(); errors are NULL / -1 with the message in ktt_last_error().
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KTT_API __attribute__((visibility("default")))
typedef void* ktt_h;

KTT_API int ktt_abi_version(void);
KTT_API const char* ktt_last_error(void);

/* ---- device */
KTT_API int ktt_open(const int* ids, int n, int rows, int cols);
KTT_API int ktt_close(void);
KTT_API int ktt_is_open(void);
KTT_API int ktt_synchronize(void);
KTT_API int ktt_mark_exiting(void);
KTT_API int ktt_mesh_size(void);
/* out[13]: uploads downloads ops op_ns programs desc_hits alloc_ns args_ns desc_ns
 * enqueue_ns program_entries kernel_compiles, (12 used). */
KTT_API int ktt_counters(int64_t* out);
KTT_API int ktt_memory(int64_t* out2); /* allocated, peak */
KTT_API int ktt_clear_program_cache(void);
KTT_API int ktt_numerics_push(int fidelity, int approx, int strict, int attention);
KTT_API int ktt_numerics_pop(void);

/* ---- handles */
KTT_API void ktt_release(ktt_h h);
KTT_API ktt_h ktt_retain(ktt_h h);
KTT_API ktt_h
ktt_upload(const void* data, const int64_t* shape, int rank, int dtype, int format);
KTT_API int ktt_download(ktt_h h, void* dst, int chip);
KTT_API ktt_h ktt_empty(const int64_t* shape, int rank, int dtype, int format);
KTT_API ktt_h ktt_clone(ktt_h h);
KTT_API int ktt_copy_into(ktt_h dst, ktt_h src);
/* Device shape: returns the rank (and writes up to cap sizes into out), or -1. */
KTT_API int ktt_shape(ktt_h h, int64_t* out, int cap);
KTT_API int ktt_dtype(ktt_h h);
KTT_API int ktt_format(ktt_h h);

/* ---- views */
KTT_API ktt_h ktt_reshape(ktt_h a, const int64_t* sizes, int n);
KTT_API ktt_h ktt_permute(ktt_h a, const int64_t* dims, int n);
KTT_API ktt_h ktt_slice(
    ktt_h a, const int64_t* begins, const int64_t* ends, const int64_t* steps, int n
);
KTT_API ktt_h ktt_concat(const ktt_h* parts, int count, int64_t dim);
KTT_API ktt_h ktt_repeat(ktt_h a, const int64_t* repeats, int n);
KTT_API ktt_h ktt_typecast(ktt_h a, int dtype);
KTT_API ktt_h ktt_bitcast(ktt_h a, int dtype);

/* ---- elementwise */
KTT_API ktt_h ktt_unary(int op, ktt_h a);
KTT_API ktt_h ktt_binary(int op, ktt_h a, ktt_h b);
KTT_API ktt_h ktt_binary_scalar(int op, ktt_h a, double s);
KTT_API ktt_h ktt_where(ktt_h cond, ktt_h a, ktt_h b);
KTT_API ktt_h ktt_clamp(ktt_h a, int has_lo, double lo, int has_hi, double hi);
KTT_API ktt_h ktt_full(const int64_t* shape, int rank, double value, int dtype);

/* ---- random draws on the device; equal seeds (keys) give equal draws. */
/* U(from, to) shaped and typed like `like`. */
KTT_API ktt_h ktt_uniform(ktt_h like, double from, double to, uint32_t seed);
/* N(0, 1). */
KTT_API ktt_h ktt_normal(const int64_t* shape, int rank, int dtype, uint32_t seed);
/* Philox4x32-10 uniforms in [0, 1): word `word` (0..3) of the counter at each
 * element's index under `key`. */
KTT_API ktt_h ktt_philox_uniform(
    const int64_t* shape, int rank, int dtype, uint64_t key, uint32_t word
);

/* ---- reductions / index. `dims` are device dims, `out` the logical sizes of the
 * result. */
KTT_API ktt_h ktt_sum(ktt_h a, const int64_t* dims, int nd, const int64_t* out, int no);
KTT_API ktt_h
ktt_amax(ktt_h a, const int64_t* dims, int nd, const int64_t* out, int no);
KTT_API ktt_h ktt_softmax(ktt_h a, int64_t dim);
KTT_API ktt_h ktt_argmax_last(ktt_h a, const int64_t* out, int no);
KTT_API ktt_h ktt_cumsum(ktt_h a, int64_t dim);
KTT_API ktt_h
ktt_embedding_rows(ktt_h table, int64_t version, ktt_h index, int64_t count);
KTT_API ktt_h ktt_gather(ktt_h a, int64_t dim, ktt_h index);

/* ---- matmul */
KTT_API ktt_h
ktt_mm(ktt_h a, ktt_h b, int ta, int tb, int out_dtype /* -1: default */, int fast);
KTT_API ktt_h ktt_linear(ktt_h x, ktt_h w, ktt_h bias /* may be NULL */, int kn);

/* ---- fused ops. Multi-result functions write handles into `out` (NULL entries for
 * absent results). */
KTT_API int ktt_layer_norm(
    ktt_h x, int64_t normalized, double eps, ktt_h gamma, ktt_h beta, ktt_h* out3
);
KTT_API int ktt_layer_norm_backward(
    ktt_h grad,
    ktt_h x,
    ktt_h mean,
    ktt_h rstd,
    int64_t normalized,
    ktt_h gamma,
    int want_input,
    int want_gamma,
    int want_beta,
    ktt_h* out3
);
KTT_API int ktt_rms_norm_train(ktt_h x, double eps, ktt_h gamma, ktt_h* out3);
KTT_API int ktt_rms_norm_backward(
    ktt_h grad,
    ktt_h x,
    ktt_h rstd,
    ktt_h gamma,
    int want_input,
    int want_gamma,
    ktt_h* out3
);
KTT_API ktt_h ktt_rms_norm(ktt_h x, double eps, ktt_h gamma, ktt_h beta);
KTT_API ktt_h ktt_sdpa(
    ktt_h q, ktt_h k, ktt_h v, ktt_h mask, int causal, int has_scale, double scale
);
KTT_API int ktt_sdpa_train(ktt_h q, ktt_h k, ktt_h v, int causal, ktt_h* out2);
KTT_API int ktt_sdpa_train_backward(
    ktt_h grad,
    ktt_h out,
    ktt_h q,
    ktt_h k,
    ktt_h v,
    ktt_h stats,
    int causal,
    ktt_h* out3
);
KTT_API ktt_h ktt_conv2d(
    ktt_h x,
    ktt_h w,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
);
KTT_API ktt_h ktt_conv2d_input_grad(
    ktt_h go,
    ktt_h w,
    const int64_t* xshape4,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
);
KTT_API ktt_h ktt_conv2d_weight_grad(
    ktt_h go,
    ktt_h x,
    const int64_t* wshape4,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
);
/* params: beta1 beta2 scale eps step decay l2; out3 = p', m', v' (returns -2 when the
 * inputs do not qualify). */
KTT_API int ktt_adam_step(
    ktt_h p, ktt_h g, ktt_h m, ktt_h v, const double* params7, int negate, ktt_h* out3
);
/* `a` with `src` written along device dim `dim` at int32 `index` (same rank; index
 * sizes at most src's, and a's off `dim`); reduce 0 replace, 1 add, 2 multiply. */
KTT_API ktt_h ktt_scatter(ktt_h a, int64_t dim, ktt_h index, ktt_h src, int reduce);

#ifdef __cplusplus
}
#endif
