FieldScout - Wheat Disease Detection from Field Images | PyTorch, ConvNeXt, Grad-CAM, ONNX Runtime, C++, OpenCV
- Built a 5-class wheat disease classifier (yellow rust, brown rust, septoria, mildew, healthy) on 999 field images; improved macro-F1 from 0.88 (EfficientNet-B0) to ~0.96 (ConvNeXt-Tiny), validated across two independent leakage-safe splits with bootstrap confidence intervals
- Used Grad-CAM to detect background shortcut learning, and uncovered a camera confound in the public dataset (metadata alone predicted disease at 0.60 macro-F1); showed the model held ~0.97 accuracy within single-camera subsets
- Exported to ONNX and wrote C++ inference (ONNX Runtime + OpenCV) matching Python on 100% of test images; benchmarked INT8 quantisation trade-offs (4x smaller, slower on CPU)
- Prototyped a weather-driven early-warning layer (live forecast data, risk scoring, alerting with cooldown, photo confirmation, SQLite event timeline)
