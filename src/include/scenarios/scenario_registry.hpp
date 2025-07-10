/**
 * @file    scenario_registry.hpp
 * @brief   Factory and registry for all benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Provides dynamic registration and creation of scenarios by name.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include "scenario_interface.hpp"

 /**
  * @class   ScenarioRegistry
  * @brief   Factory and registry for all available scenarios.
  *
  * Allows dynamic registration and creation of scenarios by name.
  */
class ScenarioRegistry {
public:
	/// Type for scenario creator function.
	using Creator = std::function<std::unique_ptr<ScenarioInterface>()>;

	/**
	 * @brief Singleton accessor.
	 * @return Reference to the global ScenarioRegistry.
	 */
	static ScenarioRegistry& instance() {
		static ScenarioRegistry registry;
		return registry;
	}

	/**
	 * @brief Register a scenario creator for a specific scenario name.
	 *
	 * @param name    Name of the scenario (e.g. "dummy").
	 * @param creator Functor that returns a new scenario instance.
	 */
	void register_scenario(const std::string& name, Creator creator) {
		creators_[name] = std::move(creator);
	}

	/**
	 * @brief Create a new instance of a registered scenario.
	 *
	 * @param name Name of the scenario to create.
	 * @return     Unique pointer to the created scenario, or nullptr if not found.
	 */
	std::unique_ptr<ScenarioInterface> create(const std::string& name) const {
		auto it = creators_.find(name);
		if (it != creators_.end())
			return (it->second)();
		return nullptr;
	}

	/**
	 * @brief Get a list of all registered scenario names.
	 * @return Vector of registered scenario names.
	 */
	std::vector<std::string> available_scenarios() const {
		std::vector<std::string> result;
		for (const auto& kv : creators_) result.push_back(kv.first);
		return result;
	}

private:
	std::unordered_map<std::string, Creator> creators_; ///< Map: scenario name -> factory function
};