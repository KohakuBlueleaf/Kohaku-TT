// PrivateUse1 hooks for the autograd engine and generators: tt is built and available,
// device 0 has a primary context, and the default generator is a host-side generator
// tagged with the tt device (random kernels draw on host and upload).
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace ktt {
    namespace {

        // The CPU generator algorithm, owned by tt:0.
        struct TtGeneratorImpl : at::CPUGeneratorImpl {
            TtGeneratorImpl() {
                device_ = c10::Device(c10::DeviceType::PrivateUse1, 0);
                key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
            }
        };

        struct TtHooks final : at::PrivateUse1HooksInterface {
            bool isBuilt() const override {
                return true;
            }

            bool isAvailable() const override {
                return true;
            }

            const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
                static auto generator = at::make_generator<TtGeneratorImpl>();
                return generator;
            }

            at::Generator getNewGenerator(c10::DeviceIndex) const override {
                return at::make_generator<TtGeneratorImpl>();
            }

            at::Device getDeviceFromPtr(void*) const override {
                return c10::Device(c10::DeviceType::PrivateUse1, 0);
            }

            bool hasPrimaryContext(c10::DeviceIndex) const override {
                return true;
            }
        };

        const bool registered = [] {
            at::RegisterPrivateUse1HooksInterface(new TtHooks());
            return true;
        }();

    }  // namespace
}  // namespace ktt
