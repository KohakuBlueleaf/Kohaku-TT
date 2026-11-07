// aten random kernels for tt, drawn on device (seeded per call from the tt generator;
// no host draw or upload): distributions, random_, bernoulli, normal, dropout, rrelu.
// multinomial and full-range random_ draw on host.
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>

#include <cmath>
#include <mutex>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // The tt generator `gen` (or the default one).
        at::Generator tt_generator(std::optional<at::Generator> gen) {
            return (
                gen ? *gen
                    : at::globalContext().defaultGenerator(
                          c10::Device(c10::DeviceType::PrivateUse1, 0)
                      )
            );
        }

        // The next 32-bit device seed from the tt generator (never 0).
        uint32_t next_seed(std::optional<at::Generator> gen) {
            auto tt = tt_generator(gen);
            std::lock_guard<std::mutex> lock(tt.mutex());
            auto* impl = (
                static_cast<at::CPUGeneratorImpl*>(tt.unsafeGetGeneratorImpl())
            );
            return static_cast<uint32_t>(impl->random()) | 1u;
        }

        // A CPU generator seeded from the tt generator (host-drawn ops only).
        at::Generator host_generator(std::optional<at::Generator> gen) {
            auto tt = tt_generator(gen);
            std::lock_guard<std::mutex> lock(tt.mutex());
            auto* impl = (
                static_cast<at::CPUGeneratorImpl*>(tt.unsafeGetGeneratorImpl())
            );
            return at::detail::createCPUGenerator(impl->random64());
        }

        // The floating dtype a draw for `self` runs in (bf16 stays bf16, others fp32).
        at::ScalarType draw_dtype(const at::Tensor& self) {
            return self.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kFloat;
        }

        // U(from, to) on device, sizes of `self`, in draw_dtype(self).
        at::Tensor uniform_like(
            const at::Tensor& self,
            double from,
            double to,
            std::optional<at::Generator> gen
        ) {
            auto dtype = draw_dtype(self);
            auto like = at::empty(self.sizes(), self.options().dtype(dtype));
            auto h = bridge::uniform(dense("uniform_", like), from, to, next_seed(gen));
            return wrap_output(h, self.sizes(), dtype);
        }

        // N(0, 1) on device of `sizes` in `dtype`.
        at::Tensor standard_normal(
            at::IntArrayRef sizes,
            at::ScalarType dtype,
            std::optional<at::Generator> gen
        ) {
            auto h = bridge::normal(to_vector(sizes), to_bridge(dtype), next_seed(gen));
            return wrap_output(h, sizes, dtype);
        }

        // Bernoulli(p) on device per element of probabilities `p` (0 / 1 in p's dtype).
        at::Tensor bernoulli_of(const at::Tensor& p, std::optional<at::Generator> gen) {
            auto h = bridge::bernoulli(dense("bernoulli", p), next_seed(gen));
            return wrap_output(h, p.sizes(), p.scalar_type());
        }

        at::Tensor& normal_(
            at::Tensor& self, double mean, double std, std::optional<at::Generator> gen
        ) {
            auto noise = standard_normal(self.sizes(), draw_dtype(self), gen);
            return (
                write_into("normal_", self, (noise * std + mean).to(self.scalar_type()))
            );
        }

        at::Tensor& uniform_(
            at::Tensor& self, double from, double to, std::optional<at::Generator> gen
        ) {
            auto u = uniform_like(self, from, to, gen);
            return write_into("uniform_", self, u.to(self.scalar_type()));
        }

        at::Tensor& bernoulli_(
            at::Tensor& self, double p, std::optional<at::Generator> gen
        ) {
            auto probs = (
                at::full(self.sizes(), p, self.options().dtype(draw_dtype(self)))
            );
            return (
                write_into(
                    "bernoulli_.float",
                    self,
                    bernoulli_of(probs, gen).to(self.scalar_type())
                )
            );
        }

        // Bernoulli with per-element probabilities p (broadcast to self).
        at::Tensor& bernoulli_tensor_(
            at::Tensor& self, const at::Tensor& p, std::optional<at::Generator> gen
        ) {
            auto probs = p.to(draw_dtype(self)).expand(self.sizes()).contiguous();
            auto result = bernoulli_of(probs, gen).to(self.scalar_type());
            return write_into("bernoulli_.Tensor", self, result);
        }

        // Inverse-CDF transforms of u ~ U(0, 1) for the other distributions.
        at::Tensor& exponential_(
            at::Tensor& self, double lambd, std::optional<at::Generator> gen
        ) {
            auto u = uniform_like(self, 0.0, 1.0, gen).to(at::kFloat);
            auto values = -at::log1p(-u) / lambd;
            return write_into("exponential_", self, values.to(self.scalar_type()));
        }

        at::Tensor& cauchy_(
            at::Tensor& self,
            double median,
            double sigma,
            std::optional<at::Generator> gen
        ) {
            auto u = uniform_like(self, 0.0, 1.0, gen).to(at::kFloat);
            auto values = at::tan((u - 0.5) * M_PI) * sigma + median;
            return write_into("cauchy_", self, values.to(self.scalar_type()));
        }

        at::Tensor& geometric_(
            at::Tensor& self, double p, std::optional<at::Generator> gen
        ) {
            auto u = uniform_like(self, 0.0, 1.0, gen).to(at::kFloat);
            auto values = at::ceil(at::log1p(-u) / std::log1p(-p));
            return write_into("geometric_", self, values.to(self.scalar_type()));
        }

        at::Tensor& log_normal_(
            at::Tensor& self, double mean, double std, std::optional<at::Generator> gen
        ) {
            auto noise = standard_normal(self.sizes(), at::kFloat, gen);
            return (
                write_into(
                    "log_normal_", self, at::exp(noise * std + mean).to(self.scalar_type())
                )
            );
        }

        // Integers in [from, to) as floor(U(from, to)) on device; ranges past 2^24
        // (fp32 exact integers) draw on host.
        at::Tensor& random_from_(
            at::Tensor& self,
            int64_t from,
            std::optional<int64_t> to,
            std::optional<at::Generator> gen
        ) {
            constexpr int64_t kExact = int64_t{1} << 24;
            if (!to || *to - from > kExact) {
                auto values = at::empty(self.sizes(), self.options().device(at::kCPU));
                values.random_(from, to, host_generator(gen));
                return write_into("random_.from", self, values.to(self.device()));
            }
            auto like = at::empty(self.sizes(), self.options().dtype(at::kFloat));
            auto u = (
                wrap_output(
                    bridge::uniform(
                        dense("random_.from", like),
                        static_cast<double>(from),
                        static_cast<double>(*to),
                        next_seed(gen)
                    ),
                    self.sizes(),
                    at::kFloat
                )
            );
            auto values = at::clamp(at::floor(u), static_cast<double>(from), *to - 1.0);
            return write_into("random_.from", self, values.to(self.scalar_type()));
        }

        at::Tensor& random_to_(
            at::Tensor& self, int64_t to, std::optional<at::Generator> gen
        ) {
            return random_from_(self, 0, to, gen);
        }

        // The dtype's full range, as PyTorch: drawn on host.
        at::Tensor& random_full_(at::Tensor& self, std::optional<at::Generator> gen) {
            auto values = at::empty(self.sizes(), self.options().device(at::kCPU));
            values.random_(host_generator(gen));
            return write_into("random_", self, values.to(self.device()));
        }

        at::Tensor normal_tensor_float(
            const at::Tensor& mean, double std, std::optional<at::Generator> gen
        ) {
            auto noise = standard_normal(mean.sizes(), at::kFloat, gen);
            return (mean.to(at::kFloat) + noise * std).to(mean.scalar_type());
        }

        at::Tensor normal_float_tensor(
            double mean, const at::Tensor& std, std::optional<at::Generator> gen
        ) {
            auto noise = standard_normal(std.sizes(), at::kFloat, gen);
            return (std.to(at::kFloat) * noise + mean).to(std.scalar_type());
        }

        at::Tensor normal_tensor_tensor(
            const at::Tensor& mean,
            const at::Tensor& std,
            std::optional<at::Generator> gen
        ) {
            auto shape = at::infer_size(mean.sizes(), std.sizes());
            auto noise = standard_normal(shape, at::kFloat, gen);
            auto out = mean.to(at::kFloat) + std.to(at::kFloat) * noise;
            return out.to(at::result_type(mean, std));
        }

        // (input * keep / (1 - p), keep) with keep ~ Bernoulli(1 - p) on device;
        // identity and an all-true mask when not training or p == 0.
        std::tuple<at::Tensor, at::Tensor> native_dropout(
            const at::Tensor& input, double p, std::optional<bool> train
        ) {
            if ((train.has_value() && !*train) || p == 0.0) {
                auto all = at::ones(input.sizes(), input.options().dtype(at::kBool));
                return {input.clone(), all};
            }
            auto probs = (
                at::full(
                    input.sizes(), 1.0 - p, input.options().dtype(draw_dtype(input))
                )
            );
            auto keep = bernoulli_of(probs, std::nullopt);
            auto mask = keep.to(at::kBool);
            if (p == 1.0) {
                return {at::zeros_like(input), mask};
            }
            return {input * keep.to(input.scalar_type()) * (1.0 / (1.0 - p)), mask};
        }

        at::Tensor native_dropout_backward(
            const at::Tensor& grad, const at::Tensor& mask, double scale
        ) {
            return grad * mask * scale;
        }

        // Randomized leaky relu: training draws slopes U(lower, upper) for x < 0 into
        // noise (1 elsewhere); eval uses the mean slope.
        at::Tensor rrelu_with_noise(
            const at::Tensor& self,
            at::Tensor& noise,
            const at::Scalar& lower,
            const at::Scalar& upper,
            bool training,
            std::optional<at::Generator> gen
        ) {
            const double lo = lower.toDouble(), hi = upper.toDouble();
            if (!training) {
                return at::where(at::ge(self, 0), self, self * ((lo + hi) / 2.0));
            }
            auto draws = uniform_like(self, lo, hi, gen).to(self.scalar_type());
            auto slopes = at::where(at::ge(self, 0), at::ones_like(self), draws);
            write_into("rrelu_with_noise", noise, slopes.to(noise.scalar_type()));
            return self * slopes;
        }

        // grad * noise (training, a real range), else the leaky relu gradient of the
        // mean slope.
        at::Tensor rrelu_with_noise_backward(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& noise,
            const at::Scalar& lower,
            const at::Scalar& upper,
            bool training,
            bool
        ) {
            const double lo = lower.toDouble(), hi = upper.toDouble();
            if (training && hi - lo > 1e-6) {
                return grad * noise;
            }
            return at::where(at::gt(self, 0), grad, grad * ((lo + hi) / 2.0));
        }

        // Category draws from row probabilities `self` (1-D or 2-D), on host.
        at::Tensor multinomial(
            const at::Tensor& self,
            int64_t num_samples,
            bool replacement,
            std::optional<at::Generator> gen
        ) {
            auto host = self.to(at::kFloat).cpu();
            auto picks = (
                at::multinomial(host, num_samples, replacement, host_generator(gen))
            );
            return picks.to(self.device());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("normal_", &normal_);
        m.impl("uniform_", &uniform_);
        m.impl("bernoulli_.float", &bernoulli_);
        m.impl("exponential_", &exponential_);
        m.impl("cauchy_", &cauchy_);
        m.impl("geometric_", &geometric_);
        m.impl("log_normal_", &log_normal_);
        m.impl("random_.from", &random_from_);
        m.impl("random_.to", &random_to_);
        m.impl("random_", &random_full_);
        m.impl("bernoulli_.Tensor", &bernoulli_tensor_);
        m.impl("normal.Tensor_float", &normal_tensor_float);
        m.impl("normal.float_Tensor", &normal_float_tensor);
        m.impl("normal.Tensor_Tensor", &normal_tensor_tensor);
        m.impl("native_dropout", &native_dropout);
        m.impl("native_dropout_backward", &native_dropout_backward);
        m.impl("multinomial", &multinomial);
        m.impl("rrelu_with_noise", &rrelu_with_noise);
        m.impl("rrelu_with_noise_backward", &rrelu_with_noise_backward);
    }

}  // namespace ktt
