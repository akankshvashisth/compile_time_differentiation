#ifndef __defines_hpp__
#define __defines_hpp__

#include <iostream>
#define AKS_FUNCTION_PREFIX_ATTR // __device__ __host__

namespace aks {
	namespace assert_detail {
		template<typename T>
		inline void gpu_assert(T code, const char* file, int line, bool abort = true)
		{

		}

		//inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true)
		//{
		//	if (code != cudaSuccess) {
		//		fprintf(stderr, "gpu_assert: %s %s %s(%d)\n", cudaGetErrorString(code), cudaGetErrorName(code), file, line);
		//		if (abort) {
		//			exit(code);
		//		}
		//	}
		//}
	}
}

#define gpu_error_check(code) { ::aks::assert_detail::gpu_assert((code), __FILE__, __LINE__); }

#endif // !__defines_hpp__