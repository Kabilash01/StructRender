#include "structrender/engine.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iterator>
#include <sstream>

namespace structrender {

size_t TensorInfo::elements() const {
    size_t n = 1;
    for (long d : shape) n *= static_cast<size_t>(d);
    return n;
}

namespace {
class Logger : public nvinfer1::ILogger {
public:
    std::string last;
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR) last = msg;
    }
};
}  // namespace

struct Engine::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    std::vector<TensorInfo> tensors;
    std::vector<void*> dev;
    cudaStream_t stream = nullptr;
    std::string err;

    bool fail(const std::string& m) {
        err = m + (logger.last.empty() ? "" : " | trt: " + logger.last);
        return false;
    }
    bool cuda(cudaError_t e, const char* what) {
        if (e == cudaSuccess) return true;
        return fail(std::string(what) + ": " + cudaGetErrorString(e));
    }
    void release() {
        for (void* d : dev) if (d) cudaFree(d);
        dev.clear();
        ctx.reset(); engine.reset(); runtime.reset();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        tensors.clear();
    }
};

Engine::Engine() : p_(new Impl) {}
Engine::~Engine() { p_->release(); }
bool Engine::loaded() const { return p_->ctx != nullptr; }
const std::vector<TensorInfo>& Engine::tensors() const { return p_->tensors; }
const std::string& Engine::last_error() const { return p_->err; }

bool Engine::load(const std::string& path) {
    p_->release();
    std::ifstream f(path, std::ios::binary);
    if (!f) return p_->fail("cannot open engine " + path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    p_->runtime.reset(nvinfer1::createInferRuntime(p_->logger));
    if (!p_->runtime) return p_->fail("createInferRuntime failed");
    p_->engine.reset(p_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!p_->engine) return p_->fail("deserializeCudaEngine failed (engine built by another TensorRT?)");
    p_->ctx.reset(p_->engine->createExecutionContext());
    if (!p_->ctx) return p_->fail("createExecutionContext failed");
    if (!p_->cuda(cudaStreamCreate(&p_->stream), "cudaStreamCreate")) return false;
    for (int i = 0; i < p_->engine->getNbIOTensors(); ++i) {
        TensorInfo t;
        t.name = p_->engine->getIOTensorName(i);
        t.is_input = p_->engine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        if (p_->engine->getTensorDataType(t.name.c_str()) != nvinfer1::DataType::kFLOAT)
            return p_->fail("tensor " + t.name + " is not float32 (FP16 engines take float32 I/O here)");
        auto d = p_->engine->getTensorShape(t.name.c_str());
        for (int k = 0; k < d.nbDims; ++k) {
            if (d.d[k] < 0) return p_->fail("tensor " + t.name + " has a dynamic dimension; fixed shapes only (§12)");
            t.shape.push_back(d.d[k]);
        }
        void* buf = nullptr;
        if (!p_->cuda(cudaMalloc(&buf, t.elements() * sizeof(float)), "cudaMalloc")) return false;
        p_->dev.push_back(buf);
        if (!p_->ctx->setTensorAddress(t.name.c_str(), buf)) return p_->fail("setTensorAddress " + t.name);
        p_->tensors.push_back(t);
    }
    p_->err.clear();
    return true;
}

bool Engine::infer(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) {
    if (!loaded()) return p_->fail("infer before load");
    size_t ii = 0, oi = 0;
    for (size_t k = 0; k < p_->tensors.size(); ++k) {
        const auto& t = p_->tensors[k];
        if (!t.is_input) continue;
        if (ii >= inputs.size() || !inputs[ii]) return p_->fail("missing input " + t.name);
        if (!p_->cuda(cudaMemcpyAsync(p_->dev[k], inputs[ii++], t.elements() * sizeof(float),
                                      cudaMemcpyHostToDevice, p_->stream), "H2D")) return false;
    }
    if (!p_->ctx->enqueueV3(p_->stream)) return p_->fail("enqueueV3 failed");
    for (size_t k = 0; k < p_->tensors.size(); ++k) {
        const auto& t = p_->tensors[k];
        if (t.is_input) continue;
        if (oi >= outputs.size() || !outputs[oi]) return p_->fail("missing output buffer " + t.name);
        if (!p_->cuda(cudaMemcpyAsync(outputs[oi++], p_->dev[k], t.elements() * sizeof(float),
                                      cudaMemcpyDeviceToHost, p_->stream), "D2H")) return false;
    }
    return p_->cuda(cudaStreamSynchronize(p_->stream), "sync");
}

bool Engine::benchmark(int warmup, int iters, std::vector<float>& ms) {
    if (!loaded()) return p_->fail("benchmark before load");
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < warmup; ++i) p_->ctx->enqueueV3(p_->stream);
    cudaStreamSynchronize(p_->stream);
    ms.clear();
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a, p_->stream);
        if (!p_->ctx->enqueueV3(p_->stream)) return p_->fail("enqueueV3 failed in benchmark");
        cudaEventRecord(b, p_->stream);
        cudaEventSynchronize(b);
        float t = 0; cudaEventElapsedTime(&t, a, b); ms.push_back(t);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return true;
}

}  // namespace structrender
