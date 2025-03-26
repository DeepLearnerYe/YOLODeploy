# 环境
ubuntu: 18.04  
GNU C Library: 2.27  
CUDA: 11.8.0  
TensorRT: 8.6.1.6  
OpenCV: 4.5.4  
# 结构
```

├── include/                     # 头文件
│   ├── core/                    # 核心框架模块
│   │   ├── base_model.hpp       # 模型抽象基类定义
│   │   ├── base_model.hpp       # 模板实现文件
│   │   ├── backend.hpp          # 推理后端抽象接口定义
│   │   └── types.hpp            # 公共数据类型定义
│   ├── models/                  # 具体模型
│   │   ├── yolov11_det.hpp      # YOLOv11检测模型
│   │   ├── yolov11_cls.hpp      # YOLOv11分类模型
│   ├── backends/                # 推理后端实现
│   │   ├── atlas_backend.hpp    # Atlas后端
│   │   ├── tensorrt_backend.hpp # TensorRT推理后端
├── src/                         # 源码
├── java/                        # JNI接口实现目录
├── examples/                    # 示例代码目录
    ├── yolov11_singlePicture.cpp 

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