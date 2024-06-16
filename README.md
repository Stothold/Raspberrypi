Project Overview
This project aims to create a Blackjack system using a Raspberry Pi, camera, and LED indicators. The system operates in a loop with the following steps:

Capture card images using a PiCamera.
Detect cards using YOLOv5 and convert the data into a list.
Calculate Blackjack strategy (stand if the total is 17 or higher, hit if less).
Light up LEDs based on the decision.
The system also focuses on optimization to reduce CPU usage and heat generation, thus improving battery life and portability.

Performance Metrics
Execution time per iteration.
CPU and memory usage (measured using the built-in task manager).
Temperature (logged from /sys/class/thermal).
Results
Implementation
Video of the working system [link to video].
Optimization
Performance: Reduced execution time by 49%.
Efficiency: Reduced heat generation by 34% while maintaining execution time within 1 second.
Limitations and Solutions
Implementation
Challenges
The main challenge was the project environment, which includes hardware limitations and the complexities of working with a Raspberry Pi.

Solution
Python was used for its readability and extensive libraries, allowing for rapid prototyping and development.

Optimization
Challenges
Model Performance Optimization: Low accuracy in card detection.
Raspberry Pi Hardware Constraints: Limited capability to run PyTorch-based models efficiently.
ONNX Limitations: Fixed output of 25,200 bounding boxes requiring extensive post-processing.
Python Limitations: Performance bottlenecks in computationally intensive tasks.
Solutions
Model Architecture and Training:

Switched from S model to N model with more epochs for better accuracy.
Training setup: 20000 images, batch size 8, 200 epochs.
Framework Optimization:

Utilized ONNX Runtime for better CPU efficiency.
Implemented graph optimizations to reduce unnecessary operations and enhance execution efficiency.
Non-Maximum Suppression (NMS):

Applied NMS to efficiently remove duplicate bounding boxes.
Binary Code Conversion:

Converted critical Python code to C libraries to enhance performance.
Reflection
Future work could involve leveraging more of C++ to overcome Python's performance limitations, especially for computationally intensive tasks.
