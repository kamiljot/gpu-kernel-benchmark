/**
 * @file    dummy_inference_scenario.cpp
 * @brief   Example ML scenario for demonstrating modular benchmarking workflow.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Implements a dummy scenario combining backend, kernel, and profiler for demonstration purposes.
 */

#include <iostream>
#include <memory>
#include <vector>

#include "../backend/cuda_backend.hpp"
#include "../kernels/sin_cos_pow_relu/sin_cos_pow_relu_kernel.hpp"
#include "../profiler/cuda_profiler.hpp"
#include "scenario.hpp"

 /**
  * @class   DummyInferenceScenario
  * @brief   Example scenario that launches a kernel and profiles execution.
  */
class DummyInferenceScenario : public Scenario
{
public:
	DummyInferenceScenario(CudaBackend* backend, Kernel* kernel) : backend_(backend), kernel_(kernel)
	{
	}

	std::string name() const override
	{
		return "dummy_inference";
	}

	void setup() override
	{
		// Setup input data and allocate output vector
		input_a.assign(1024, 1.0f);
		input_b.assign(1024, 2.0f);
		output.assign(1024, 0.0f);
		std::cout << "[DummyInferenceScenario] Setup complete.\n";
	}

	void run() override
	{
		CudaProfiler profiler;
		profiler.start();
		backend_->launch(kernel_, input_a, input_b, output);
		profiler.stop();
		std::cout << "[DummyInferenceScenario] Kernel executed in " << profiler.elapsed_time_ms() << " ms.\n";
		profiler.save_to_json("dummy_scenario_profile.json");
	}

	void teardown() override
	{
		input_a.clear();
		input_b.clear();
		output.clear();
		std::cout << "[DummyInferenceScenario] Resources cleaned up.\n";
	}

private:
	CudaBackend* backend_;
	Kernel* kernel_;
	std::vector<float> input_a, input_b, output;
};