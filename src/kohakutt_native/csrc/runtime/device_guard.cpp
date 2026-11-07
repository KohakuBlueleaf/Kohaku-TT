// The PrivateUse1 device guard: one device (tt:0) with one default stream.
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace ktt {
    namespace {

        struct TtGuardImpl final : c10::impl::DeviceGuardImplInterface {
            c10::DeviceType type() const override {
                return c10::DeviceType::PrivateUse1;
            }

            c10::Device exchangeDevice(c10::Device) const override {
                return c10::Device(c10::DeviceType::PrivateUse1, 0);
            }

            c10::Device getDevice() const override {
                return c10::Device(c10::DeviceType::PrivateUse1, 0);
            }

            void setDevice(c10::Device) const override {
            }

            void uncheckedSetDevice(c10::Device) const noexcept override {
            }

            c10::Stream getStream(c10::Device d) const noexcept override {
                return c10::Stream(c10::Stream::DEFAULT, d);
            }

            c10::Stream exchangeStream(c10::Stream s) const noexcept override {
                return s;
            }

            c10::DeviceIndex deviceCount() const noexcept override {
                return 1;
            }
        };

    }  // namespace

    C10_REGISTER_GUARD_IMPL(PrivateUse1, TtGuardImpl);

}  // namespace ktt
