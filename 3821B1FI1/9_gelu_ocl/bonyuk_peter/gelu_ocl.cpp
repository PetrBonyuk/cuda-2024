﻿/*When entering the following vector:
std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0};
The output values ​​were:
0.841192 1.9546 2.99636 3.99993 5
*/

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>

std::vector<float> GeluOCL(const std::vector<float> &input) {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms.front();

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cl::Device device = devices.front();

	cl::Context context(device);
	cl::CommandQueue queue(context);

	std::string codeKernel = R"(
__kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
  int i = get_global_id(0);

  if (i < countElem) {
      float val = x[i];
      float tmp = val * (1.595769122f + val * val * 0.071354816f);
      y[i] = val - val / (1.0f + exp(tmp));
  }
}
)";

	cl::Program::Sources sources;
	sources.emplace_back(std::move(codeKernel));
	cl::Program program(context, sources);

	if (program.build() != CL_SUCCESS) {
		std::cerr << "\033[1;31merror\033[0m: ";
		std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
		return {};
	}


	if (input.empty()) return {};
	auto size = input.size();
	auto countBytes = size * sizeof(float);

	cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, countBytes);
	cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, countBytes);

	queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, countBytes, input.data());

	cl::Kernel kernel(program, "gelu_kernel");
	kernel.setArg(0, bufferInput);
	kernel.setArg(1, bufferOutput);
	kernel.setArg(2, static_cast<int>(size));
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	std::vector<float> output(size);
	queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, countBytes, output.data());

	return output;
}