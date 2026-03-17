#include "fastcuda/runtime.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string operator_name = "gemm";
    std::string kernel_name = "baseline";
    std::string shape = "m=1024,n=1024,k=1024";
    std::string dtype = "fp16";

    if (argc > 1) {
        operator_name = argv[1];
    }
    if (argc > 2) {
        kernel_name = argv[2];
    }
    if (argc > 3) {
        shape = argv[3];
    }
    if (argc > 4) {
        dtype = argv[4];
    }

    try {
        std::cout << "operator=" << operator_name << "\n";
        std::cout << "kernel=" << kernel_name << "\n";
        std::cout << "shape=" << shape << "\n";
        std::cout << "dtype=" << dtype << "\n";
        std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());
        std::cout << "status=benchmark_stub\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error=" << ex.what() << "\n";
        return 1;
    }
}
