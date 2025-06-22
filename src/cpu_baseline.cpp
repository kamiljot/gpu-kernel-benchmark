#include <cmath>

// CPU reference implementation of: c[i] = sqrt(a[i]) + log(b[i])
void cpu_add(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) {
        c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f);
    }
}
