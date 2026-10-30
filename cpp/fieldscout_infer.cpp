// FieldScout C++ inference: wheat disease classification with ONNX Runtime + OpenCV.
//
// Usage:
//   ./fieldscout_infer model.onnx classes.txt image1.jpg [image2.jpg ...]
//   ./fieldscout_infer model.onnx classes.txt --list images.txt
//
// Prints one CSV line per image: path,predicted_class,confidence,inference_ms
// Preprocessing matches the Python eval transform:
//   resize shorter side to int(size * 1.14) -> centre crop size x size -> RGB -> /255 -> ImageNet normalise

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3] = {0.229f, 0.224f, 0.225f};

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

// Returns a CHW float tensor of shape 3 x size x size.
std::vector<float> preprocess(const cv::Mat& bgr, int size) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Resize so the shorter side equals int(size * 1.14), keeping aspect ratio (as torchvision does).
    const int short_target = static_cast<int>(size * 1.14);
    const int h = rgb.rows, w = rgb.cols;
    int new_h, new_w;
    if (h <= w) {
        new_h = short_target;
        new_w = static_cast<int>(short_target * static_cast<double>(w) / h);
    } else {
        new_w = short_target;
        new_h = static_cast<int>(short_target * static_cast<double>(h) / w);
    }
    const bool shrinking = new_h < h;
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, shrinking ? cv::INTER_AREA : cv::INTER_LINEAR);

    // Centre crop.
    const int top = static_cast<int>(std::round((new_h - size) / 2.0));
    const int left = static_cast<int>(std::round((new_w - size) / 2.0));
    cv::Mat crop = resized(cv::Rect(left, top, size, size));

    // HWC uint8 -> CHW float, normalised.
    std::vector<float> chw(3 * size * size);
    for (int y = 0; y < size; ++y) {
        const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                chw[c * size * size + y * size + x] = (row[x][c] / 255.0f - kMean[c]) / kStd[c];
            }
        }
    }
    return chw;
}

std::vector<float> softmax(const float* logits, size_t n) {
    std::vector<float> p(logits, logits + n);
    const float m = *std::max_element(p.begin(), p.end());
    float sum = 0.f;
    for (auto& v : p) { v = std::exp(v - m); sum += v; }
    for (auto& v : p) v /= sum;
    return p;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " model.onnx classes.txt image.jpg [...] | --list images.txt\n";
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        const std::vector<std::string> classes = read_lines(argv[2]);

        std::vector<std::string> images;
        if (std::string(argv[3]) == "--list" && argc >= 5) images = read_lines(argv[4]);
        else for (int i = 3; i < argc; ++i) images.emplace_back(argv[i]);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fieldscout");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);  // single core, like a small edge device
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = session.GetInputNameAllocated(0, alloc);
        auto out_name = session.GetOutputNameAllocated(0, alloc);
        const char* in_names[] = {in_name.get()};
        const char* out_names[] = {out_name.get()};

        // Read the expected image size from the model: shape is [batch, 3, size, size].
        auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const int size = static_cast<int>(shape[2]);
        const std::vector<int64_t> input_shape = {1, 3, size, size};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::cout << "path,predicted_class,confidence,inference_ms\n";
        for (const auto& path : images) {
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) { std::cerr << "skipping unreadable image: " << path << "\n"; continue; }

            std::vector<float> input = preprocess(img, size);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                                input_shape.data(), input_shape.size());

            auto t0 = std::chrono::steady_clock::now();
            auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
            auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            const float* logits = outputs[0].GetTensorData<float>();
            std::vector<float> probs = softmax(logits, classes.size());
            const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();

            std::cout << path << "," << classes[best] << "," << probs[best] << "," << ms << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
