/**
 * @file    scenario.hpp
 * @brief   Abstract scenario interface for ML benchmarking workflows.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares the Scenario interface for ML benchmarking workflows and batch runs.
 */

#pragma once

#include <string>

 /**
  * @class   Scenario
  * @brief   Abstract interface for an ML benchmarking scenario.
  */
class Scenario
{
public:
	virtual ~Scenario()
	{
	}

	/**
	 * @brief Returns the scenario's name.
	 */
	virtual std::string name() const = 0;

	/**
	 * @brief Scenario setup: allocates resources, loads data, etc.
	 */
	virtual void setup() = 0;

	/**
	 * @brief Runs the scenario (calls kernels/benchmarks).
	 */
	virtual void run() = 0;

	/**
	 * @brief Cleans up resources after scenario completion.
	 */
	virtual void teardown() = 0;
};