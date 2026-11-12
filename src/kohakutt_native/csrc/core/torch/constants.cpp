// Device constants cache: one tt tensor per key, built from host data on first use.
#include "core/torch/constants.hpp"

#include <map>
#include <mutex>

namespace ktt {
    namespace {

        std::mutex g_mutex;

        // Never destroyed at exit (after TT-Metal's teardown); close() clears it.
        std::map<std::string, at::Tensor>& constants() {
            static auto* map = new std::map<std::string, at::Tensor>();
            return *map;
        }

    }  // namespace

    at::Tensor device_constant(
        const std::string& key, const std::function<at::Tensor()>& build
    ) {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto& table = constants();
        auto it = table.find(key);
        if (it != table.end()) {
            return it->second;
        }
        auto value = build().to(c10::Device(c10::DeviceType::PrivateUse1, 0));
        table.emplace(key, value);
        return value;
    }

    bool has_device_constant(const std::string& key) {
        std::lock_guard<std::mutex> lock(g_mutex);
        return constants().count(key) != 0;
    }

    void clear_device_constants() {
        std::lock_guard<std::mutex> lock(g_mutex);
        constants().clear();
    }

}  // namespace ktt
