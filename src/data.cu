#include "src/data.h"

#include <algorithm>
#include <cmath>
#include <random>

void generate_fp16(half* h_A, half* h_B, int M, int N, int K, int num_batches) {
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

	const size_t a_count = static_cast<size_t>(num_batches) * static_cast<size_t>(M) * static_cast<size_t>(K);
	const size_t b_count = static_cast<size_t>(num_batches) * static_cast<size_t>(K) * static_cast<size_t>(N);

	for (size_t i = 0; i < a_count; ++i) {
		h_A[i] = __float2half(dist(rng));
	}
	for (size_t i = 0; i < b_count; ++i) {
		h_B[i] = __float2half(dist(rng));
	}
}

void cpu_gemm_fp16(const half* A, const half* B, float* C, int M, int N, int K) {
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float acc = 0.0f;
			for (int k = 0; k < K; ++k) {
				const float a = __half2float(A[static_cast<size_t>(m) * K + k]);
				const float b = __half2float(B[static_cast<size_t>(k) * N + n]);
				acc += a * b;
			}
			C[static_cast<size_t>(m) * N + n] = acc;
		}
	}
}

bool verify(const float* ref, const float* out, int M, int N, float tol) {
	const size_t count = static_cast<size_t>(M) * static_cast<size_t>(N);
	for (size_t i = 0; i < count; ++i) {
		const float r = ref[i];
		const float o = out[i];
		const float allowed = tol * std::fabs(r) + tol;
		if (std::fabs(r - o) > allowed) {
			return false;
		}
	}
	return true;
}

AccuracyResult measure_accuracy(const float* ref, const float* out, int M, int N, float pass_tol) {
	const size_t count = static_cast<size_t>(M) * static_cast<size_t>(N);

	float max_abs = 0.0f;
	double sum_sq = 0.0;
	double rel_pct_sum = 0.0;

	for (size_t i = 0; i < count; ++i) {
		const double r = static_cast<double>(ref[i]);
		const double o = static_cast<double>(out[i]);
		const double diff = std::fabs(r - o);

		max_abs = std::max(max_abs, static_cast<float>(diff));
		sum_sq += diff * diff;

		const double denom = std::max(std::fabs(r), 1e-8);
		rel_pct_sum += (diff / denom) * 100.0;
	}

	AccuracyResult result{};
	result.max_abs_err = max_abs;
	result.rmse = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
	result.real_err_pct = static_cast<float>(rel_pct_sum / static_cast<double>(count));
	result.pass = verify(ref, out, M, N, pass_tol);
	return result;
}
