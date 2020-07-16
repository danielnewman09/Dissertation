NOTE: All of these papers are formatted for submission to the SME Journal of Manufacturing Processes

# Paper 1

__An Open-Source Data Acquisition and Statistical Inference Device for Manufacturing Health Monitoring__

Daniel Newman, Thomas Kurfess, Andrew Dugenske

_In recent years, substantial research interest has been directed toward data acquisition and machine learning within the context of manufacturing health monitoring. To deploy such immensely useful technologies at scale, the hardware and software used to implement them must be reliable and cost-effective. To this end, recent developments have enabled advanced data acquisition and processing technology to be deployed with open-source, royalty-free hardware and software. This paper describes the design of an integrated data acquisition device which leverages these state-of-the-art, open-source tools. Relying on the BeagleBone platform, this data acquisition device leverages real-time-capable hardware and advanced machine learning software to incorporate sensor data acquisition and analysis in a single device. This same strategy is applied to CNC controller data as well, allowing an interface to MTConnect and OPC-UA protocols._

## Background

- Industrial Internet of Things: Why do we want to do all of this data acquisition and machine learning? 
- Discuss proprietary data and its drawbacks. This affects devices from existing companies such as National Instruments and others.
- Discuss available and researched data acquisition devices. These devices each have drawbacks which can be addressed by increasingly powerful existing technology. 
- Discuss the importance of utilizing existing hardware and software instead of developing new devices from scratch

## Materials and Methods

- Node Red: Controller and Analog Sensor Data Acquisition
- BeagleBone Black
- ADXL Accelerometer

## Results

### Analog Data Acquisition 
- Show Signal Data Acquisition results in capturing a simple sinusoid

### Computational Latency
- Show onboard latency results for accelerometer data

### Emco Warm-Up Data
- Show Warm-up program data, both from the controller and the extracted frequency content
- These results will show how rapidly this device is capable of extracting high-quality frequency content alongside CNC controller data


# Paper 2

__Contextual Acquisition of Machine Health and Process Data in an IoT Framework__

The advancement of the internet of things has led to new data acquisition and health monitoring strategies for manufacturing equipment. To leverage these technologies at scale, health data must be contextualized within current machine and process status. This paper introduces a strategy for automatic data contextualization utililzing existing messaging standards in a 3-layer IoT architecture. This approach requires the concurrent acquisition of machine process and health data --- 


This chapter describes the methodology and design of an architecture for transmitting machine data from a factory floor to cloud storage for analysis and historical monitoring. Specifically, this architecture leverages modern web protocols such as MQTT to enable low-overhead bidirectional communication for data acquisition devices in a factory setting. Using a standardized messaging format, communication across a variety of machines may be facilitated. This architecture also incorporates the use of additional sensors for health monitoring. By leveraging CNC controller data to indicate the current operational state of a machine, sensor data are readily contextualized relative to machine status. Two simple case studies illustrate the usefulness of sensor data in detecting tool wear and tracking spindle health in a milling machine.

## Background

## Materials and Methods

- Reference Data Acquisition device


## Results

- Show Warm-Up RMS trends over time
- Show Example data of warm-up program
- 


# Paper 3

__Edge vs Cloud: A Comparison of Feature Extraction and Machine Learning Inference Approaches__

## Background

- Include discussion on the advancement of embedded computing for statistical inference
- Benefits of edge computing include increased security and reduced network bandwidth. Be sure to flesh these benefits out fully
- Open-Source Statistical and Machine Learning tools

## Materials and Methods

- Compare Feature Extraction Methods


# Paper 4

__A Tool Wear Classification Case Study for Edge Data Acquisition and Machine Learning Inference__








