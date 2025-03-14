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