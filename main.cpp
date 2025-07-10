/**
 * @file    main.cpp
 * @brief   Entry point for the GPU Kernel Benchmark project. Supports modular kernel selection and is ready for future ML scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "kernels/add/add_kernel.hpp"
#include "kernels/sin_cos_pow_relu/sin_cos_pow_relu_kernel.hpp"
 // Future: #include "scenarios/ml_inference/ml_inference_scenario.hpp"

 /**
  * @brief Parses CLI arguments. Supports --op and --size.
  */
void parse_cli(int argc, char* argv[], std::string& op, int& size)
{
	// Default values
	op = "add";
	size = 1024;

	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg == "--op" && i + 1 < argc)
		{
			op = argv[++i];
		}
		else if (arg == "--size" && i + 1 < argc)
		{
			size = std::stoi(argv[++i]);
		}
		// Future: Add more CLI options (e.g., --scenario, --variant, --input)
	}
}

/**
 * @brief Main entry point. Selects and launches the chosen kernel.
 */
int main(int argc, char* argv[])
{
	std::string op;
	int N;

	parse_cli(argc, argv, op, N);

	std::vector<float> a(N, 1.0f);      ///< Example input vector a
	std::vector<float> b(N, 2.0f);      ///< Example input vector b
	std::vector<float> out(N, 0.0f);    ///< Output vector

	std::unique_ptr<Kernel> kernel;

	if (op == "add")
	{
		kernel = std::make_unique<AddKernel>();
	}
	else if (op == "sin_cos_pow_relu")
	{
		kernel = std::make_unique<SinCosPowReluKernel>();
	}
	// Future: else if (op == "ml_inference") { kernel = std::make_unique<MLInferenceScenario>(); }
	else
	{
		std::cerr << "Unknown operation: " << op << std::endl;
		std::cerr << "Available operations: add, sin_cos_pow_relu" << std::endl;
		return 1;
	}

	kernel->launch(a, b, out);

	std::cout << "[GPU Kernel Benchmark] Operation: " << kernel->name() << ", N = " << N << std::endl;
	std::cout << "First output value: " << out[0] << std::endl;

	// Future: Profile results, save to file, run ML scenarios etc.

	return 0;
}