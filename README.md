# 环境
ubuntu: 18.04
GNU C Library: 2.27
CUDA: 11.8.0
TensorRT: 8.6.1.6
OpenCV: 4.5.4
# 结构
```

├── include/
│   ├── core/              // 核心框架接口
│   │   ├── base_model.hpp // 模型基类
│   │   ├── backend.hpp    // 后端接口
│   │   ├── exceptions.hpp
│   │   └── types.hpp      // 公共数据结构
│   ├── models/           // 具体模型实现
│   │   ├── yolov5_det.hpp
│   │   ├── yolov11_det.hpp
│   │   ├── yolov11_cls.hpp
│   ├── backends/         // 推理后端实现
│   │   ├── atlas_backend.hpp
│   │   ├── tensorrt_backend.hpp
├── src/

└── samples/
    ├── detection_demo.cpp
    └── classification_demo.cpp
```

# 模型转换
```
# 路径是存放模型的路径，需要有模型的.engine和.names文件
./convert_model.sh <path/to/model>
```
转换的模型在脚本文件最开始的`models`指定，支持多个模型一起转换

# TODO
1. make a utils directory by sperating function of reading file in `Initialize()`
2. seperate logger