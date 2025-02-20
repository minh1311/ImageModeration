#include "TRTRunner.hpp"

#include <iostream>
#include <fstream>

TRTRunner::TRTRunner(const std::string& strModelPath, Options_t& stOptions)
{
    m_stOptions = stOptions;
    // get onnxModelPath file extension
    std::string strModelPathExt = strModelPath.substr(strModelPath.find_last_of(".") + 1);
    if (strModelPathExt == "onnx")
    {
        m_strEngineName = serializeEngineOptions(strModelPath);
    }
    else if (strModelPathExt == "engine")
    {
        m_strEngineName = strModelPath;
    }
    else
    {
        // throw std::runtime_error(std::string(TagTRTRunner) + "Error, model file must be either an .onnx or .engine file!");
        error(TagTRTRunner, "Error, model file must be either an .onnx or .engine file!");
        return ;
    }

    std::string strPathEngine = strModelPath.substr(0, strModelPath.find_last_of("/") + 1) + m_strEngineName;

    // Set the device index
    auto ret = cudaSetDevice(m_stOptions.deviceIndex);
    if (ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_stOptions.deviceIndex) 
                        + ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        // throw std::runtime_error(std::string(TagTRTRunner) + errMsg);
        error(TagTRTRunner, errMsg);
        return ;
    }

    if (doesFileExist(strPathEngine))
    {
        bool rc = loadEngine(strPathEngine);
        if (!rc)
        {
            // throw std::runtime_error(std::string(TagTRTRunner) + "Failed to load TRT engine.");
            error(TagTRTRunner, "Failed to load TRT engine.");
            return ;
        }
    }
    else
    {
        bool rc = buildEngine(strModelPath);
        if (!rc)
        {
            // throw std::runtime_error(std::string(TagTRTRunner) + "Failed to build TRT engine.");
            error(TagTRTRunner, "Failed to build TRT engine.");
            return ;
        }
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context)
    {
        // throw std::runtime_error(std::string(TagTRTRunner) + "Error, create context failed!");
        error(TagTRTRunner, "Error, create context failed!");
    }

    cudaStreamCreate(&m_stream);

    // std::cout << TagTRTRunner << "Created TRTRunner for " << strModelPath << " successfully!" << std::endl;
    info(TagTRTRunner, "Created TRTRunner for " + strModelPath + " successfully!");
}

TRTRunner::~TRTRunner()
{
    cudaStreamDestroy(m_stream);
}

void TRTRunner::allocateIOBuffers(std::vector<void*>& buffers, std::unordered_map<std::string, std::vector<size_t>>& umpIOTensors, 
                                    std::unordered_map<std::string, nvinfer1::Dims>& umpIOTensorsShape)
{
    const int iNbIOTensors = m_engine->getNbIOTensors();
    buffers.resize(iNbIOTensors);

    // Allocate GPU memory for input and output buffers
    for (size_t i = 0; i < iNbIOTensors; i++)
    {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
    
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        // auto tensorDtype = m_engine->getTensorDataType(tensorName);
        nvinfer1::Dims tensorShape = m_engine->getTensorShape(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT)
        {
            tensorShape.d[0] = m_stOptions.optBatchSize;
            tensorShape.d[2] = m_stOptions.optHeight;
            tensorShape.d[3] = m_stOptions.optWidth;

            // m_context->setInputShape(tensorName, tensorShape);
            // m_context->setBindingDimensions(i, tensorShape);
            #if NV_TENSORRT_MAJOR >= 10
                m_context->setInputShape(tensorName, tensorShape);
            #else
                m_context->setBindingDimensions(i, tensorShape);
            #endif

        }
        else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // int iIndex = m_engine->getBindingIndex(tensorName);
            // tensorShape = m_context->getBindingDimensions(i);
            #if NV_TENSORRT_MAJOR >= 10
                tensorShape = m_context->getTensorShape(tensorName);
            #else
                tensorShape = m_context->getBindingDimensions(i);
            #endif

        }
        else
        {
            // throw std::runtime_error(std::string(TagTRTRunner) + "Error, IO Tensor is neither an input or output!");
            error(TagTRTRunner, "Error, IO Tensor is neither an input or output!");
            return ;
        }

        size_t tensorSize = getSizeByDims(tensorShape);

        checkCudaErrorCode(cudaMallocAsync(&buffers[i], tensorSize * sizeof(float), m_stream));

        m_umpIOTensors[tensorName] = {i, tensorSize};
        m_umpIOTensorsShape[tensorName] = tensorShape;
    }

    umpIOTensors = m_umpIOTensors;
    umpIOTensorsShape = m_umpIOTensorsShape;

    // Set the address of the input and output buffers
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), buffers[i]);
        if (!status)
        {
            // throw std::runtime_error(std::string(TagTRTRunner) + "Error setTensorAddress");
            error(TagTRTRunner, "Error setTensorAddress");
            return ;
        }
    }
}

void TRTRunner::runModel(std::vector<void*>& buffers)
{
    bool status = m_context->enqueueV3(m_stream);
    if (!status)
    {
        // throw std::runtime_error(std::string(TagTRTRunner) + "Error enqueueV3");
        error(TagTRTRunner, "Error enqueueV3");
    }
}

std::string TRTRunner::serializeEngineOptions(const std::string& strModelPath)
{
    const auto filenamePos = strModelPath.find_last_of('/') + 1;
    std::string strEngineName = strModelPath.substr(filenamePos, strModelPath.find_last_of('.') - filenamePos);

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> strDeviceNames = getDeviceNames();

    if (static_cast<size_t>(m_stOptions.deviceIndex) >= strDeviceNames.size()) {
        // throw std::runtime_error(std::string(TagTRTRunner) + "Error, provided device index is out of range!");
        error(TagTRTRunner, "Error, provided device index is out of range!");
        return "";
    }

    auto strDeviceName = strDeviceNames[m_stOptions.deviceIndex];

    // Remove spaces from the device name
    strDeviceName.erase(std::remove_if(strDeviceName.begin(), strDeviceName.end(), ::isspace), strDeviceName.end());
    strEngineName += "." + strDeviceName + ".engine";

    return strEngineName;
}

std::vector<std::string> TRTRunner::getDeviceNames()
{
    std::vector<std::string> deviceNames;
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }

    return deviceNames;
}

bool TRTRunner::buildEngine(const std::string& strModelPath)
{
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_TRTLogger));
    if (!builder)
    {
        // std::cerr << TagTRTRunner << "Error, unable to create builder!" << std::endl;
        error(TagTRTRunner, "Error, unable to create builder!");
        return false;
    }

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        // std::cerr << TagTRTRunner << "Error, unable to create network!" << std::endl;
        error(TagTRTRunner, "Error, unable to create network!");
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        // std::cerr << TagTRTRunner << "Error, unable to create builder config" << std::endl;
        error(TagTRTRunner, "Error, unable to create builder config");
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_TRTLogger));
    if (!parser)
    {
        // std::cerr << TagTRTRunner << "Error, unable to create parser!" << std::endl;
        error(TagTRTRunner, "Error, unable to create parser!");
        return false;
    }

    bool ret = parser->parseFromFile(strModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!ret)
    {
        // std::cerr << TagTRTRunner << "Failure while parsing ONNX file!" << std::endl;
        error(TagTRTRunner, "Failure while parsing ONNX file!");
        return false;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 3_GiB);
    // std::cout << config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE) << std::endl;

    // set optimization profile. Note: Only support 1 input
    nvinfer1::IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    
    const auto numInputs = network->getNbInputs();
    for (int i = 0; i < numInputs; i++)
    {
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();

        if (inputDims.d[0] == -1)
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, 
                                        nvinfer1::Dims4(1, inputDims.d[1], m_stOptions.optHeight, m_stOptions.optWidth));
        else
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, 
                                        nvinfer1::Dims4(inputDims.d[0], inputDims.d[1], m_stOptions.optHeight, m_stOptions.optWidth));

        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, 
                                    nvinfer1::Dims4(m_stOptions.optBatchSize, inputDims.d[1], m_stOptions.optHeight, m_stOptions.optWidth));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, 
                                    nvinfer1::Dims4(m_stOptions.maxBatchSize, inputDims.d[1], m_stOptions.maxHeight, m_stOptions.maxWidth));
    }
    config->addOptimizationProfile(optProfile);

    if (m_stOptions.precision == Precision_t::FP16)
    {
        if (!builder->platformHasFastFp16())
        {
            // std::cerr << TagTRTRunner << "Error: GPU does not support FP16 precision!" << std::endl;
            error(TagTRTRunner, "Error: GPU does not support FP16 precision!");
            return false;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    // cudaStreamCreate(&profileStream);
    config->setProfileStream(profileStream);

    // m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    // Write the engine to disk
    std::string strOutEngineFile = strModelPath.substr(0, strModelPath.find_last_of("/") + 1) + m_strEngineName;
    // std::cout << TagTRTRunner << "Generating engine: " << strOutEngineFile << " ..." << std::endl;
    info(TagTRTRunner, "Generating engine: " + strOutEngineFile + " ...");

    // save engine
    // nvinfer1::IHostMemory *plan = m_engine->serialize();
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        // std::cout << TagTRTRunner << "Error, unable to build engine!" << std::endl;
        error(TagTRTRunner, "Error, unable to build engine!");
        return false;
    }
    std::ofstream outfile;
    outfile.open(strOutEngineFile, std::ios::binary | std::ios::out);
    outfile.write((const char *) plan->data(), plan->size());
    outfile.close();
    // delete plan;

    // std::cout << TagTRTRunner << "Success, saved engine to " << strOutEngineFile << std::endl;
    info(TagTRTRunner, "Success, saved engine to " + strOutEngineFile);

    // network->destroy();
    // config->destroy();
    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    // cudaStreamDestroy(profileStream);

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime> {nvinfer1::createInferRuntime(m_TRTLogger)};
    if (!m_runtime)
    {
        // std::cerr << TagTRTRunner << "Error, createInferRuntime failed!" << std::endl;
        error(TagTRTRunner, "Error, createInferRuntime failed!");
        return false;
    }
    
    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!m_engine)
    {
        // std::cerr << TagTRTRunner << "Error, deserializeCudaEngine failed!" << std::endl;
        error(TagTRTRunner, "Error, deserializeCudaEngine failed!");
        return false;
    }

    // std::cout << TagTRTRunner << "Load engine: " << strOutEngineFile << " successfully!" << std::endl;
    info(TagTRTRunner, "Load engine: " + strOutEngineFile + " successfully!");

    return true;
}

bool TRTRunner::loadEngine(std::string& strPathEngine)
{
    // Read the serialized model from disk
    std::ifstream file(strPathEngine, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        // std::cerr << TagTRTRunner << "Error, Unable to read engine file!" << std::endl;
        error(TagTRTRunner, "Error, Unable to read engine file!");
        return false;
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime> {nvinfer1::createInferRuntime(m_TRTLogger)};
    if (!m_runtime)
    {
        // std::cerr << TagTRTRunner << "Error, createInferRuntime failed!" << std::endl;
        error(TagTRTRunner, "Error, createInferRuntime failed!");
        return false;
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine)
    {
        // std::cerr << TagTRTRunner << "Error, deserializeCudaEngine failed!" << std::endl;
        error(TagTRTRunner, "Error, deserializeCudaEngine failed!");
        return false;
    }

    // std::cout << TagTRTRunner << "Load " << strPathEngine << " successfully!" << std::endl;
    info(TagTRTRunner, "Load " + strPathEngine + " successfully!");
    
    return true;
}

size_t TRTRunner::getSizeByDims(const nvinfer1::Dims& tensorShape)
{
    size_t size = 1;
    for (size_t i = 0; i < tensorShape.nbDims; i++)
    {
        size *= tensorShape.d[i];
    }
    return size;
}
