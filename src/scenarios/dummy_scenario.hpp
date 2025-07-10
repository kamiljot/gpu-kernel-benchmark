/**
 * @file    dummy_scenario.hpp
 * @brief   Example dummy scenario for demonstration/testing.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Shows how to implement and register a simple scenario.
 */

#pragma once

#include "../include/scenarios/scenario_interface.hpp"

 /**
  * @class   DummyScenario
  * @brief   Example scenario that does nothing useful.
  */
class DummyScenario : public ScenarioInterface {
public:
	DummyScenario();

	std::string name() const override;
	void configure(const std::vector<std::pair<std::string, std::string>>& params) override;
	void run() override;
	std::string result() const override;

private:
	int dummy_param_;
	std::string result_;
};