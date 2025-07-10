#include <iostream>
#include <vector>
#include <cassert>

#include "kernels/add/add_kernel.hpp"

/**
 * @brief Minimal test for AddKernel.
 */
int main() {
	const int N = 8;
	std::vector<float> a(N), b(N), out(N, 0.0f);

	// Fill test data
	for (int i = 0; i < N; ++i) {
		a[i] = static_cast<float>(i);
		b[i] = 2.0f * i;
	}

	AddKernel kernel;
	kernel.launch(a, b, out);

	// Check result
	for (int i = 0; i < N; ++i) {
		float expected = a[i] + b[i];
		if (std::abs(out[i] - expected) > 1e-5f) {
			std::cerr << "Test failed at i = " << i << ": expected " << expected << ", got " << out[i] << std::endl;
			return 1;
		}
	}

	std::cout << "AddKernel test PASSED." << std::endl;
	return 0;
}