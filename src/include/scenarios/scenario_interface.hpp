/**
 * @file    scenario_interface.hpp
 * @brief   Abstract interface for all benchmark scenarios (batch, ML, etc.)
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Defines the common interface for every benchmark scenario.
 */

#pragma once

#include <string>
#include <vector>

 /**
  * @class   ScenarioInterface
  * @brief   Base interface for all benchmark scenarios.
  *
  * Each scenario represents a high-level benchmarking use-case,
  * such as "batch inference", "matrix multiply", etc.
  */
class ScenarioInterface {
public:
	virtual ~ScenarioInterface() = default;

	/**
	 * @brief Get the name of the scenario.
	 * @return Scenario name (e.g. "dummy", "ml_batch", "gemm")
	 */
	virtual std::string name() const = 0;

	/**
	 * @brief Configure the scenario with input parameters.
	 *
	 * @param params Arbitrary key-value parameters (can be empty).
	 */
	virtual void configure(const std::vector<std::pair<std::string, std::string>>& params) = 0;

	/**
	 * @brief Run the benchmark scenario.
	 *
	 * Should run any logic, possibly using kernels from the registry.
	 */
	virtual void run() = 0;

	/**
	 * @brief Get the result of the scenario as a string (e.g. CSV line, JSON, summary).
	 * @return String with scenario result.
	 */
	virtual std::string result() const = 0;
};