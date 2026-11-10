// C ABI over ktt::bridge::* for the tinygrad backend (see ktt_capi.h), built into
// _ktt_native.so with the bridge sources; no torch headers.
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "core/device/tt_bridge.hpp"
#include "py_bridge/ktt_capi.h"

namespace br = ktt::bridge;

namespace {

    struct Box {
        br::HandlePtr p;
    };

    thread_local std::string g_error;

    using Shape = std::vector<int64_t>;

    Shape vec(const int64_t* p, int n) {
        return p && n > 0 ? Shape(p, p + n) : Shape();
    }

    br::HandlePtr hp(ktt_h h) {
        return h ? static_cast<Box*>(h)->p : nullptr;
    }

    ktt_h box(br::HandlePtr p) {
        return p ? new Box{std::move(p)} : nullptr;
    }

    br::DType dt(int code) {
        if (code < 0 || code > 4) {
            throw std::invalid_argument(
                "ktt_capi: bad dtype code " + std::to_string(code)
            );
        }
        return static_cast<br::DType>(code);
    }

    br::Format fmt(int code) {
        if (code < 0 || code > 2) {
            throw std::invalid_argument(
                "ktt_capi: bad format code " + std::to_string(code)
            );
        }
        return static_cast<br::Format>(code);
    }

    std::array<int64_t, 2> two(const int64_t* p) {
        return {p[0], p[1]};
    }

    template <typename F>
    ktt_h guard_h(F f) {
        try {
            return box(f());
        } catch (const std::exception& e) {
            g_error = e.what();
        } catch (...) {
            g_error = "unknown C++ exception";
        }
        return nullptr;
    }

    template <typename F>
    int guard_i(F f) {
        try {
            return f();
        } catch (const std::exception& e) {
            g_error = e.what();
        } catch (...) {
            g_error = "unknown C++ exception";
        }
        return -1;
    }

}  // namespace

extern "C" {

int ktt_abi_version(void) {
    return 1;
}

const char* ktt_last_error(void) {
    return g_error.c_str();
}

int ktt_open(const int* ids, int n, int rows, int cols) {
    return guard_i([&] {
        br::open(std::vector<int>(ids, ids + n), rows, cols);
        return 0;
    });
}

int ktt_close(void) {
    return guard_i([] {
        br::close();
        return 0;
    });
}

int ktt_is_open(void) {
    return br::is_open() ? 1 : 0;
}

int ktt_synchronize(void) {
    return guard_i([] {
        br::synchronize();
        return 0;
    });
}

int ktt_mark_exiting(void) {
    br::mark_exiting();
    return 0;
}

int ktt_mesh_size(void) {
    return guard_i([] {
        return br::mesh_size();
    });
}

int ktt_counters(int64_t* out) {
    return guard_i([&] {
        const auto c = br::counters();
        const int64_t v[] = {
            c.uploads,
            c.downloads,
            c.ops,
            c.op_ns,
            c.programs,
            c.desc_hits,
            c.alloc_ns,
            c.args_ns,
            c.desc_ns,
            c.enqueue_ns,
            c.program_entries,
            c.kernel_compiles
        };
        std::memcpy(out, v, sizeof(v));
        return 0;
    });
}

int ktt_memory(int64_t* out2) {
    return guard_i([&] {
        const auto m = br::memory_stats();
        out2[0] = m.allocated;
        out2[1] = m.peak;
        return 0;
    });
}

int ktt_clear_program_cache(void) {
    return guard_i([] {
        br::clear_program_cache();
        return 0;
    });
}

int ktt_numerics_push(int fidelity, int approx, int strict, int attention) {
    return guard_i([&] {
        br::numerics_push(fidelity, approx != 0, strict != 0, attention);
        return 0;
    });
}

int ktt_numerics_pop(void) {
    return guard_i([] {
        br::numerics_pop();
        return 0;
    });
}

void ktt_release(ktt_h h) {
    delete static_cast<Box*>(h);
}

ktt_h ktt_retain(ktt_h h) {
    return box(hp(h));
}

ktt_h ktt_upload(
    const void* data, const int64_t* shape, int rank, int dtype, int format
) {
    return guard_h([&] {
        return br::upload(data, vec(shape, rank), dt(dtype), fmt(format));
    });
}

int ktt_download(ktt_h h, void* dst, int chip) {
    return guard_i([&] {
        br::download(hp(h), dst, chip);
        return 0;
    });
}

ktt_h ktt_empty(const int64_t* shape, int rank, int dtype, int format) {
    return guard_h([&] {
        return br::empty(vec(shape, rank), dt(dtype), fmt(format));
    });
}

ktt_h ktt_clone(ktt_h h) {
    return guard_h([&] {
        return br::clone(hp(h));
    });
}

int ktt_copy_into(ktt_h dst, ktt_h src) {
    return guard_i([&] {
        br::copy_into(hp(dst), hp(src));
        return 0;
    });
}

int ktt_shape(ktt_h h, int64_t* out, int cap) {
    return guard_i([&] {
        const auto s = br::device_sizes(hp(h));
        for (int k = 0; k < static_cast<int>(s.size()) && k < cap; ++k) {
            out[k] = s[k];
        }
        return static_cast<int>(s.size());
    });
}

int ktt_dtype(ktt_h h) {
    return guard_i([&] {
        return static_cast<int>(br::dtype(hp(h)));
    });
}

int ktt_format(ktt_h h) {
    return guard_i([&] {
        return static_cast<int>(br::format(hp(h)));
    });
}

ktt_h ktt_reshape(ktt_h a, const int64_t* sizes, int n) {
    return guard_h([&] {
        return br::reshape(hp(a), vec(sizes, n));
    });
}

ktt_h ktt_permute(ktt_h a, const int64_t* dims, int n) {
    return guard_h([&] {
        return br::permute(hp(a), vec(dims, n));
    });
}

ktt_h ktt_slice(
    ktt_h a, const int64_t* begins, const int64_t* ends, const int64_t* steps, int n
) {
    return guard_h([&] {
        return br::slice(hp(a), vec(begins, n), vec(ends, n), vec(steps, n));
    });
}

ktt_h ktt_concat(const ktt_h* parts, int count, int64_t dim) {
    return guard_h([&] {
        std::vector<br::HandlePtr> v;
        for (int k = 0; k < count; ++k) {
            v.push_back(hp(parts[k]));
        }
        return br::concat(v, dim);
    });
}

ktt_h ktt_repeat(ktt_h a, const int64_t* repeats, int n) {
    return guard_h([&] {
        return br::repeat(hp(a), vec(repeats, n));
    });
}

ktt_h ktt_typecast(ktt_h a, int dtype) {
    return guard_h([&] {
        return br::typecast(hp(a), dt(dtype));
    });
}

ktt_h ktt_bitcast(ktt_h a, int dtype) {
    return guard_h([&] {
        return br::bitcast(hp(a), dt(dtype));
    });
}

ktt_h ktt_unary(int op, ktt_h a) {
    return guard_h([&] {
        return br::unary(static_cast<br::UnaryOp>(op), hp(a));
    });
}

ktt_h ktt_binary(int op, ktt_h a, ktt_h b) {
    return guard_h([&] {
        return br::binary(static_cast<br::BinaryOp>(op), hp(a), hp(b));
    });
}

ktt_h ktt_binary_scalar(int op, ktt_h a, double s) {
    return guard_h([&] {
        return br::binary_scalar(static_cast<br::BinaryOp>(op), hp(a), s);
    });
}

ktt_h ktt_where(ktt_h cond, ktt_h a, ktt_h b) {
    return guard_h([&] {
        return br::where(hp(cond), hp(a), hp(b));
    });
}

ktt_h ktt_clamp(ktt_h a, int has_lo, double lo, int has_hi, double hi) {
    return guard_h([&] {
        return (br::clamp(
            hp(a),
            has_lo ? std::optional<double>(lo) : std::nullopt,
            has_hi ? std::optional<double>(hi) : std::nullopt
        ));
    });
}

ktt_h ktt_full(const int64_t* shape, int rank, double value, int dtype) {
    return guard_h([&] {
        return br::full(vec(shape, rank), value, dt(dtype));
    });
}

ktt_h ktt_uniform(ktt_h like, double from, double to, uint32_t seed) {
    return guard_h([&] {
        return br::uniform(hp(like), from, to, seed);
    });
}

ktt_h ktt_normal(const int64_t* shape, int rank, int dtype, uint32_t seed) {
    return guard_h([&] {
        return br::normal(vec(shape, rank), dt(dtype), seed);
    });
}

ktt_h ktt_philox_uniform(
    const int64_t* shape, int rank, int dtype, uint64_t key, uint32_t word
) {
    return guard_h([&] {
        auto h = (br::philox_uniform(
            vec(shape, rank), dt(dtype), key, word, br::Placement{}
        ));
        if (!h) {
            throw std::invalid_argument("ktt_philox_uniform: device rank above 8");
        }
        return h;
    });
}

ktt_h ktt_sum(ktt_h a, const int64_t* dims, int nd, const int64_t* out, int no) {
    return guard_h([&] {
        return br::sum(hp(a), vec(dims, nd), vec(out, no));
    });
}

ktt_h ktt_amax(ktt_h a, const int64_t* dims, int nd, const int64_t* out, int no) {
    return guard_h([&] {
        return br::amax(hp(a), vec(dims, nd), vec(out, no));
    });
}

ktt_h ktt_softmax(ktt_h a, int64_t dim) {
    return guard_h([&] {
        return br::softmax(hp(a), dim);
    });
}

ktt_h ktt_argmax_last(ktt_h a, const int64_t* out, int no) {
    return guard_h([&] {
        return br::argmax_last(hp(a), vec(out, no));
    });
}

ktt_h ktt_cumsum(ktt_h a, int64_t dim) {
    return guard_h([&] {
        return br::cumsum(hp(a), dim);
    });
}

ktt_h ktt_embedding_rows(ktt_h table, int64_t version, ktt_h index, int64_t count) {
    return guard_h([&] {
        return br::embedding_rows(hp(table), version, hp(index), count);
    });
}

ktt_h ktt_gather(ktt_h a, int64_t dim, ktt_h index) {
    return guard_h([&] {
        return br::gather(hp(a), dim, hp(index));
    });
}

ktt_h ktt_mm(ktt_h a, ktt_h b, int ta, int tb, int out_dtype, int fast) {
    return guard_h([&] {
        return (br::mm(
            hp(a),
            hp(b),
            ta != 0,
            tb != 0,
            out_dtype < 0 ? std::nullopt : std::optional<br::DType>(dt(out_dtype)),
            fast != 0
        ));
    });
}

ktt_h ktt_linear(ktt_h x, ktt_h w, ktt_h bias, int kn) {
    return guard_h([&] {
        return br::linear(hp(x), hp(w), hp(bias), kn != 0);
    });
}

int ktt_layer_norm(
    ktt_h x, int64_t normalized, double eps, ktt_h gamma, ktt_h beta, ktt_h* out3
) {
    return guard_i([&] {
        auto r = br::layer_norm(hp(x), normalized, eps, hp(gamma), hp(beta));
        out3[0] = box(r.y);
        out3[1] = box(r.mean);
        out3[2] = box(r.rstd);
        return 0;
    });
}

int ktt_layer_norm_backward(
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
) {
    return guard_i([&] {
        auto r = (br::layer_norm_backward(
            hp(grad),
            hp(x),
            hp(mean),
            hp(rstd),
            normalized,
            hp(gamma),
            want_input != 0,
            want_gamma != 0,
            want_beta != 0
        ));
        out3[0] = box(r.input);
        out3[1] = box(r.gamma);
        out3[2] = box(r.beta);
        return 0;
    });
}

int ktt_rms_norm_train(ktt_h x, double eps, ktt_h gamma, ktt_h* out3) {
    return guard_i([&] {
        auto r = br::rms_norm_train(hp(x), eps, hp(gamma));
        out3[0] = box(r.y);
        out3[1] = box(r.mean);
        out3[2] = box(r.rstd);
        return 0;
    });
}

int ktt_rms_norm_backward(
    ktt_h grad,
    ktt_h x,
    ktt_h rstd,
    ktt_h gamma,
    int want_input,
    int want_gamma,
    ktt_h* out3
) {
    return guard_i([&] {
        auto r = (br::rms_norm_backward(
            hp(grad), hp(x), hp(rstd), hp(gamma), want_input != 0, want_gamma != 0
        ));
        out3[0] = box(r.input);
        out3[1] = box(r.gamma);
        out3[2] = box(r.beta);
        return 0;
    });
}

ktt_h ktt_rms_norm(ktt_h x, double eps, ktt_h gamma, ktt_h beta) {
    return guard_h([&] {
        return br::rms_norm(hp(x), eps, hp(gamma), hp(beta));
    });
}

ktt_h ktt_sdpa(
    ktt_h q, ktt_h k, ktt_h v, ktt_h mask, int causal, int has_scale, double scale
) {
    return guard_h([&] {
        return (br::sdpa(
            hp(q),
            hp(k),
            hp(v),
            hp(mask),
            causal != 0,
            has_scale ? std::optional<double>(scale) : std::nullopt
        ));
    });
}

int ktt_sdpa_train(ktt_h q, ktt_h k, ktt_h v, int causal, ktt_h* out2) {
    return guard_i([&] {
        auto r = br::sdpa_train(hp(q), hp(k), hp(v), causal != 0);
        out2[0] = box(r.out);
        out2[1] = box(r.stats);
        return 0;
    });
}

int ktt_sdpa_train_backward(
    ktt_h grad,
    ktt_h out,
    ktt_h q,
    ktt_h k,
    ktt_h v,
    ktt_h stats,
    int causal,
    ktt_h* out3
) {
    return guard_i([&] {
        auto r = (br::sdpa_train_backward(
            hp(grad), hp(out), hp(q), hp(k), hp(v), hp(stats), causal != 0
        ));
        for (size_t i = 0; i < 3 && i < r.size(); ++i) {
            out3[i] = box(r[i]);
        }
        return 0;
    });
}

ktt_h ktt_conv2d(
    ktt_h x,
    ktt_h w,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
) {
    return guard_h([&] {
        return br::conv2d(hp(x), hp(w), two(stride2), two(pad2), two(dil2), groups);
    });
}

ktt_h ktt_conv2d_input_grad(
    ktt_h go,
    ktt_h w,
    const int64_t* xshape4,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
) {
    return guard_h([&] {
        return (br::conv2d_input_grad(
            hp(go), hp(w), vec(xshape4, 4), two(stride2), two(pad2), two(dil2), groups
        ));
    });
}

ktt_h ktt_conv2d_weight_grad(
    ktt_h go,
    ktt_h x,
    const int64_t* wshape4,
    const int64_t* stride2,
    const int64_t* pad2,
    const int64_t* dil2,
    int64_t groups
) {
    return guard_h([&] {
        return (br::conv2d_weight_grad(
            hp(go), hp(x), vec(wshape4, 4), two(stride2), two(pad2), two(dil2), groups
        ));
    });
}

int ktt_adam_step(
    ktt_h p, ktt_h g, ktt_h m, ktt_h v, const double* params7, int negate, ktt_h* out3
) {
    return guard_i([&] {
        br::AdamStep s;
        s.beta1 = params7[0];
        s.beta2 = params7[1];
        s.scale = params7[2];
        s.eps = params7[3];
        s.step = params7[4];
        s.decay = params7[5];
        s.l2 = params7[6];
        s.negate = negate != 0;
        auto r = br::adam_step(hp(p), hp(g), hp(m), hp(v), s);
        if (r.size() < 3) {
            return -2;
        }
        for (int i = 0; i < 3; ++i) {
            out3[i] = box(r[i]);
        }
        return 0;
    });
}

}  // extern "C"
