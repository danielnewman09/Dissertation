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







