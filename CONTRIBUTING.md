# Contributing

Useful contributions include support for another ESP32-S3 board, clearer setup instructions, display improvements, and faster inference.

For a bug report, include your board, SD connection, build settings, the command you ran, and the error. Remove device serial numbers and personal paths from logs. For a hardware port, include the pin map.

Keep changes focused. Run the C and Python tests in [Development](docs/development.md). If a change affects the screen, try it on the device. If it affects performance, compare the same input and model before and after the change.

Model weights are downloaded separately. Preserve third-party license notices. Contributions to the project code and documentation are made under the MIT license.
