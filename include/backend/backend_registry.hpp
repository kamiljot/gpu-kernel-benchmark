/**
 * @file    backend_registry.hpp
 * @brief   Factory and registry for all compute backends.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Allows dynamic registration and creation of backends by name.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend_interface.hpp"

/**
 * @class   BackendRegistry
 * @brief   Factory and registry for all available compute backends.
 */
class BackendRegistry
{
   public:
    using Creator = std::function<std::unique_ptr<BackendInterface>()>;

    /**
     * @brief Singleton accessor.
     * @return Reference to the global BackendRegistry.
     */
    static BackendRegistry& instance()
    {
        static BackendRegistry registry;
        return registry;
    }

    /**
     * @brief Register a backend creator for a specific name.
     * @param name    Backend name (e.g. "cpu", "cuda").
     * @param creator Functor that returns a new backend instance.
     */
    void register_backend(const std::string& name, Creator creator)
    {
        creators_[name] = std::move(creator);
    }

    /**
     * @brief Create a new instance of a registered backend.
     * @param name Backend name.
     * @return Unique pointer to backend, or nullptr if not found.
     */
    std::unique_ptr<BackendInterface> create(const std::string& name) const
    {
        auto it = creators_.find(name);
        if (it != creators_.end()) return (it->second)();
        return nullptr;
    }

    /**
     * @brief Get a list of all registered backend names.
     * @return Vector of backend names.
     */
    std::vector<std::string> available_backends() const
    {
        std::vector<std::string> result;
        for (const auto& kv : creators_) result.push_back(kv.first);
        return result;
    }

   private:
    std::unordered_map<std::string, Creator> creators_;  ///< Map: backend name -> factory function
};