# OpenCL Labs

## Install dependencies
- Cuda (https://developer.nvidia.com/cuda-downloads)
- Intel OpenCL runtime for CPU (https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)

## Visual Studio settings
- x64 all configurations
- С/C++ -> Общие –> Дополнительные каталоги включаемых файлов = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include
- Компоновщик -> Общие –> Дополнительные каталоги библиотек = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64
- Компоновщик -> Ввод –> Дополнительные зависимости += OpenCL.lib
