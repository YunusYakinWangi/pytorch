#pragma once

// Runtime support for extern (non-Triton) CUDA kernels in cpp_wrapper mode.
//
// Mirrors lazy_triton_compile.h but for kernels whose launch still goes
// through Python (e.g. CuTeDSL).  Two C++ entry points call into the Python
// module torch._inductor.runtime.extern_kernel_compile.

#include <torch/csrc/inductor/cpp_wrapper/common.h>

struct ExternKernelSpec {
  const char* kernel_name;
  const char* kernel_source_path;
};

struct ExternModuleState {
  PyObject* pending_kernels = nullptr;
};

static PyObject* extern_kernel_compile_module = nullptr;
static PyObject* extern_start_kernel_compile = nullptr;
static PyObject* extern_run_kernel = nullptr;

static inline void loadExternCompileFuncs() {
  if (extern_kernel_compile_module == nullptr) {
    extern_kernel_compile_module = PyImport_ImportModule(
        "torch._inductor.runtime.extern_kernel_compile");
    AOTI_TORCH_CHECK(
        extern_kernel_compile_module,
        "Failed to import torch._inductor.runtime.extern_kernel_compile");

    extern_start_kernel_compile = PyObject_GetAttrString(
        extern_kernel_compile_module, "start_kernel_compile");
    AOTI_TORCH_CHECK(
        extern_start_kernel_compile,
        "Failed to get start_kernel_compile function");

    extern_run_kernel =
        PyObject_GetAttrString(extern_kernel_compile_module, "run_kernel");
    AOTI_TORCH_CHECK(extern_run_kernel, "Failed to get run_kernel function");
  }
}

static inline PyObject* getPendingKernelsForExternModule(
    ExternModuleState* module_state) {
  AOTI_TORCH_CHECK(module_state, "Invalid extern module state");
  if (module_state->pending_kernels == nullptr) {
    module_state->pending_kernels = PyDict_New();
    AOTI_TORCH_CHECK(
        module_state->pending_kernels,
        "Failed to create extern pending kernels dict");
  }
  return module_state->pending_kernels;
}

static inline void startExternKernelCompile(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    const std::string& kernel_source_path) {
  py::gil_scoped_acquire_simple acquire;

  RAIIPyObject py_name = PyUnicode_FromString(kernel_name.c_str());
  RAIIPyObject py_source_path =
      PyUnicode_FromString(kernel_source_path.c_str());
  AOTI_TORCH_CHECK(
      py_name && py_source_path, "Failed to create Python args");

  RAIIPyObject call_args =
      PyTuple_Pack(3, pending_kernels, py_name.get(), py_source_path.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result =
      PyObject_CallObject(extern_start_kernel_compile, call_args);
  AOTI_TORCH_CHECK(result, "Failed to start extern kernel compilation");
}

static inline void startExternKernelCompilesForModule(
    ExternModuleState* module_state,
    const ExternKernelSpec* const* kernel_specs,
    size_t num_kernel_specs) {
  loadExternCompileFuncs();
  PyObject* pending_kernels =
      getPendingKernelsForExternModule(module_state);
  for (size_t i = 0; i < num_kernel_specs; ++i) {
    const ExternKernelSpec* spec = kernel_specs[i];
    AOTI_TORCH_CHECK(spec, "Invalid extern kernel spec");
    startExternKernelCompile(
        pending_kernels, spec->kernel_name, spec->kernel_source_path);
  }
}

// Launch an extern kernel by calling back into Python.
// Acquires the GIL, converts C++ args to Python objects, and calls
// run_kernel(pending_kernels, kernel_name, stream, [args...]).
template <typename... Args>
static inline void runExternKernel(
    ExternModuleState* module_state,
    const char* kernel_name,
    cudaStream_t stream,
    const Args&... kernel_args) {
  py::gil_scoped_acquire_simple acquire;

  PyObject* pending_kernels =
      getPendingKernelsForExternModule(module_state);

  RAIIPyObject py_name = PyUnicode_FromString(kernel_name);
  AOTI_TORCH_CHECK(py_name, "Failed to create kernel name string");

  RAIIPyObject py_stream = PyLong_FromVoidPtr(stream);
  AOTI_TORCH_CHECK(py_stream, "Failed to create stream object");

  constexpr size_t num_args = sizeof...(Args);
  RAIIPyObject py_args_list = PyList_New(num_args);
  AOTI_TORCH_CHECK(py_args_list, "Failed to create args list");

  size_t idx = 0;
  auto add_arg = [&py_args_list, &idx](PyObject* py_arg) {
    AOTI_TORCH_CHECK(py_arg, "Failed to convert argument");
    PyList_SetItem(py_args_list, idx++, py_arg);
  };
  int dummy[] = {0, (add_arg(convertArgToPython(kernel_args)), 0)...};
  (void)dummy;

  RAIIPyObject call_args = PyTuple_Pack(
      4,
      pending_kernels,
      py_name.get(),
      py_stream.get(),
      py_args_list.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result = PyObject_CallObject(extern_run_kernel, call_args);
  AOTI_TORCH_CHECK(result, "Failed to run extern kernel");
}
