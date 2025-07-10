/**
 * @file    dummy_scenario.cpp
 * @brief   Implements the DummyScenario class.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Example: scenario that sleeps and reports the value of a param.
 */

#include "dummy_scenario.hpp"
#include <thread>
#include <chrono>
#include <sstream>

DummyScenario::DummyScenario() : dummy_param_(0), result_() {}

std::string DummyScenario::name() const {
	return "dummy";
}

void DummyScenario::configure(const std::vector<std::pair<std::string, std::string>>& params) {
	for (const auto& kv : params) {
		if (kv.first == "sleep_ms") {
			dummy_param_ = std::stoi(kv.second);
		}
	}
}

void DummyScenario::run() {
	// Simulate a "benchmark"
	std::this_thread::sleep_for(std::chrono::milliseconds(dummy_param_));
	std::ostringstream oss;
	oss << "Slept for " << dummy_param_ << " ms";
	result_ = oss.str();
}

std::string DummyScenario::result() const {
	return result_;
}