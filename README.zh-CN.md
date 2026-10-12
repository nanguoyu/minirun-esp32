# Minirun for ESP32

一个 230M 语言模型，一张 SD 卡，一块 ESP32-S3。

[English](README.md) · [开始使用](docs/getting-started.md) · [硬件接线](docs/hardware.md) · [工作原理](docs/how-it-works.md)

Minirun 让 **LFM2.5-230M 直接在 ESP32-S3 上运行**。模型放在 SD 卡里，板子每次读一小块，两个 CPU 核心配合完成读取和计算。

连接 Mac，输入一句话，就能在 240 × 240 屏幕上看它慢慢回答。屏幕会显示当前层和每个 token 的进度。Mac 负责分词、文字排版和 USB 控制，神经网络计算都在板子上完成。

```text
You: Hello!

Minirun: Hello! How can I help you today?
```

测试板生成一个新 token 约需 **37 秒**。适合喜欢动手、想看看语言模型如何在小设备里运行的人。先试一句简短的问候。

## 让它说话

准备 ESP32-S3 EYE、8 MiB PSRAM、8 MiB Flash、一张 FAT32 microSD 卡和 USB 数据线。测试使用 YD-ESP32-S3-EYE-MB v2.2，其他开发板需要核对接线。

按[安装指南](docs/getting-started.md)完成编译、刷写和模型准备，然后运行：

```sh
source .venv/bin/activate
python tools/chat.py --prompt 'Hello!' --new-tokens 16
```

只连接一块 ESP32-S3 时会自动识别串口。省略 `--prompt` 可以在终端输入。回复同时显示在终端和板载屏幕上，按 Ctrl+C 停止。

也可以试试：

```sh
python tools/chat.py --prompt '给住在茶杯里的小机器人起个名字。'
python tools/chat.py --prompt '用一句话描述月球上的灯塔。'
```

默认最多生成 32 个 token，每次运行开始一段新对话。总上下文最多 1024 个 token，由输入和回复共同使用，用多少就计算多少。模型约 149 MB，放在 SD 卡上；对话缓存最多 24 MiB，也按页存放在 SD 卡上。

代码和文档采用 [MIT 许可证](LICENSE)，模型权重遵循单独的 [LFM Open License](third_party/LFM-Open-License-v1.0.txt)。安装细节和开发文档以英文版为准。
