# Presentation Outline

0.1. Motivation

- Advancement of embedded computing
- Integration of the Internet of Things
- High-Level, 

0.2. Image showing the entire framework

- Consistent Python libraries for use on the edge and in the cloud
- Near Real-time feature extraction and inference
- Bi-directional communication between CNC machine and sensor pack to facilitate sensor data contextualization
- Consistent Messaging Format 

0.3. Video Demonstration

- Edge Data Acquisition
- Communication between Edge Device and CNC Controller
- Near Real-Time feature extraction and model inference
- 

0.4. Contributions

  - Data management strategy for the Internet of Things. 

    - This includes a strategy for utilizing machine controller data in conjunction with sensor Data.
  
  - Edge-based model inference strategy
  - Integrated Data Acquisition Device capable of performing real-time sensor data capture and near real-time inference.

0.5. Outline
  
1.1. Timeline

  - Show the progression of embedded computing technology
  - 

1.2. Industrial Internet of Things

1.3. Data Acquisition and Analysis Tools

  - MTConnect/OPC-UA
  - Sensor Data

1.4. Advancement of Open-Source Tools

1.5. Machine Health Monitoring

2.1 - Digital Architecture for Machine Health Monitoring



2.2 - Utility of Controller Data

  - Figures:
    - Good/Bad tool
    - Part Setup
    - Controller Histogram
    - Accelerometer Histogram
    - Accelerometer FFT

2.3 - Example - Historical monitoring of spindle health

  - Linked Sensor data with CNC controller data to trigger on machine startup
  - Sent commanded spindle speeds and program names 
  - Detected an honest-to-goodness anomaly
  - Figures: 
    - Spindle Warmup Program Speed
    - RMS of vibration for spindle program
    - RMS history over the course of monitoring

3.1 - Edge-Deployable Statistical and Machine Learning Inference Tools

  - Flexibility in deploying new, better algorithms as they become available
  - Build on commonly used, open-source libraries
 
3.2 - Different Machine Learning Approaches

  - Show supervised/unsupervised learning
  - Figure:
    - Tree structure showing different methods

3.3 - Data preprocessing

  - PCA
  - Data normalization
  - Labeling if possible

3.4 - Statistical Models

  - Gaussian Naive Bayes
  - Gaussian Mixture Models

3.5 - Neural Network Models

  - How a neural network functions
  - How Convolutional Neural Networks Function
  - Autoencoders

3.6 - Example Case Study with Simulated Data

  
4.1 - Integrated Data Acquisition Device



