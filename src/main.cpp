/**
 * @file    main.cpp
 * @brief   Entry point: runs selected scenario on chosen backend with CLI parameters.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Usage:
 *   ./gpu-kernel-benchmark --backend cpu --scenario dummy --param sleep_ms=200
 *
 * Example output:
 *   [INFO] Using backend: cpu
 *   [INFO] Running scenario: dummy
 *   [RESULT] Slept for 200 ms
 */

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "include/backend/backend_registry.hpp"
#include "include/scenarios/scenario_registry.hpp"

 // Simple CLI helper
std::string get_arg_value(int argc, char* argv[], const std::string& flag, const std::string& default_value = "") {
	for (int i = 1; i < argc - 1; ++i) {
		if (std::string(argv[i]) == flag)
			return argv[i + 1];
	}
	return default_value;
}

// Parse --param name=value pairs
std::vector<std::pair<std::string, std::string>> parse_params(int argc, char* argv[]) {
	std::vector<std::pair<std::string, std::string>> params;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--param" && i + 1 < argc) {
			std::string pair = argv[i + 1];
			size_t pos = pair.find('=');
			if (pos != std::string::npos) {
				params.emplace_back(pair.substr(0, pos), pair.substr(pos + 1));
			}
		}
	}
	return params;
}

int main(int argc, char* argv[]) {
	std::string backend_name = get_arg_value(argc, argv, "--backend", "cpu");
	std::string scenario_name = get_arg_value(argc, argv, "--scenario", "dummy");
	auto params = parse_params(argc, argv);

	// Create backend
	auto backend = BackendRegistry::instance().create(backend_name);
	if (!backend) {
		std::cerr << "[ERROR] Unknown backend: " << backend_name << std::endl;
		std::cerr << "Available: ";
		for (const auto& name : BackendRegistry::instance().available_backends())
			std::cerr << name << " ";
		std::cerr << std::endl;
		return 1;
	}
	std::cout << "[INFO] Using backend: " << backend_name << std::endl;

	// Create scenario
	auto scenario = ScenarioRegistry::instance().create(scenario_name);
	if (!scenario) {
		std::cerr << "[ERROR] Unknown scenario: " << scenario_name << std::endl;
		std::cerr << "Available: ";
		for (const auto& name : ScenarioRegistry::instance().available_scenarios())
			std::cerr << name << " ";
		std::cerr << std::endl;
		return 1;
	}
	std::cout << "[INFO] Running scenario: " << scenario_name << std::endl;

	scenario->configure(params);
	scenario->run();
	std::cout << "[RESULT] " << scenario->result() << std::endl;

	return 0;
}