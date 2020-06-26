# Presentation Outline

## Introduction: (7 minutes) [6:35]

## Slide 1: Title (0:15)

Good afternoon and thank you for coming to my thesis defense. Before I begin, I want to draw your attention to the links at the bottom of this slide. I have made these slides, and the code used to generate them available online. All of the data used for this research is also available on Github.

### Slide 2: The Past Decade (2:00)

With that said, I want to begin by providing some context for this research. The work I'm about to present would have been impossible ten, five, or for some elements even two years ago. And that's because the advancements in embedded computing, machine learning, and Internet of Things have dramatically changed the technological landscape in the past decade. Many of these advancements are built on technologies first introduced in the mid 2000's. A couple of examples being the ARM 32-bit microprocessor architecture and the Numpy library. 

I want to stress that with these tools, we can do some really extraordinary things on low-power devices at little cost. 
  
### Slide 3: Research Questions (1:15)

This research is specifically focused on examining how recent technological changes can be leveraged in a machine monitoring application. To that end, I am interested in answering these questions. 

...


### Slide 4: Thesis Contributions (2:30)

Through investigating these questions, I will present an integrated Internet of Things architecture and data acquisition strategy that facilitates manufacturing machine health monitoring. Specifically, this architecture incorporates three major contributions. First, ...

...While previous work has addressed individual aspects of the work, a critical advancement from this thesis is the integration of all these elements from data acquisition to health monitoring. As a result, this work shows a truly end-to-end system which can be scaled to a manufacturing environment.

### Slide 8: Demonstration (1:00)

All of these elements are demonstrated at a high-level in this video of a simple end milling process.

### Slide 9: Outline

## Background (5 Minutes)

### Slide 10: Industrial Internet of Things

This research is based on the concept the industrial internet of things. is an important concept in improving manufacturing process efficiency, rapidly responding to changes, and preventing unscheduled downtime. This figure shows a simple heriarchy of the main components involved in this .

...

I want to focus on the data acquisition process and how meaningful information can be extracted from these devices to be sent through the industrial network. This starts with the factory machines and sensors.

### Slide 11: CNC Controller Protocols

For modern CNC machines, their controllers often come equipped with software that allows important data to be readily extracted from them.

### Slide 12: Sensor Data Acquisition

In addition to controller data, external sensors can be used to enrich the data from the machine. Using sensors such as thermocouples, accelerometers, and current sensors, highly detailed information can be captured from these devices. 

### Slide 13: Health Monitoring

These controller and sensor data are typically used for something like health monitoring. This is abroad, rich field of study with many unique applications based on the specific machinery being monitored. The essense of health monitoring is quite simple: take some data stream, which may be quite large, and find a way to compress it into some low dimensional space where a health assessment can be made. 

As an example, vibration data are often used for the purpose. Motors, pumps, and really anything with a rotating shaft generates vibration which can be used to gain insight into its health state. 

Focus on vibration.

### Slide 15: Spectral Power Approximation


## Digital Architecture for Health Monitoring (6 minutes) [0:25]

Now I'll get into the first contribution of this thesis, the integrated digital architecture. This section focuses on acquiring and parsing the types of data that are necessary for manufacturing health monitoring. Specifically, this strategy seeks to record sensor and controller data to paint a complete picture of machine health and utilization.

### Slide 19: MQTT-Based Framework [1:30]

Now, when transmitting data in an IoT infrastructure, it's important to begin with an efficient and effective messaging protocol. In this case, MQTT is an excellent candidate. This is a publish-subscribe protocol which has low overhead and uses a simple topic syntax to make subscribing to a desired data stream extremely easy. As this figure shows, ...


### Slide 20: MQTT Message Definitions 

To use this protocol in a large manufacturing environment, it's important to establish a data structure. For MQTT, this includes the topic and payload as shown in this slide.

### Slide 21: Controller/Sensor Payload Example

With this message structure established, here are two example messages - one from a CNC controller and one from an accelerometer.

### Slide 23: Contextual Data Acquisition and Training




### Slide 26: Emco Warm-Up Program
### Slide 27: Emco Warm-Up History

## Edge-Deployable ML Tools (6 minutes)

With this approach, we can capture meaningful features from a manufacturing process and use them to train statistical models to predict machine health. In this section, I want to focus on the tools at our disposal to do this, and how they can be deployed on edge devices. 

### Open Source Software
### Example Dataset
### Feature Normalization
### Deployed Models
### Model Deployment
### Full Latency Comparison: Classifiers

## Integrated Edge Device (8 minutes)

So that's awesome! We've established that we can deploy fairly large machine learning models to a low power edge device using state-of-the-art software.

Now, let's integrate this functionality into an open-source data acquisition device.


### Open Source Components

### CNC Controller Integration
### Analog Data Acquisition
### Edge Device Diagram
### Device Benchmarking
### Local Compute Latency
### Cloud Compute Latency Comparison

## Case Study (10 minutes)

The summary of the experimental setup is shown in this figure. We have an embedded computer doing data acquisition on the CNC controller, an external sensor kit gathering vibration data, the MQTT message broker, cloud storage, computing, and ultimately model inference on the edge device.

### Experimental Setup

To show some more detail for the experimental setup, here are some images taken of the machine, where you see the accelerometer mounted to the spindle and its power/signal cable routed outside of the machine.

### Experimental Setup 2

This slide shows the data acquisition device used for model training. Due to campus access limitations from COVID-19, data for training was captured using an earlier version of this IoT sensor device which uses an Arduino microcontroller. This device is tolerant to higher ADC voltages and has limited sampling capacity relative to the Beaglebone. 

Model validation was still done using the device which I just showed. With slight changes in the data acquisition device and time elapsed from model training to validation, this study can actually demonstrate the relative robustness of this health monitoring approach. 

### Experimental Parameters
### Sample Labeling
### Experimental Spectrogram
### Data Composition
### Model Selection
### Hyperparameter Optimization
### Model Validation
### Model Performance: Validation
### Control Chart Analysis: Range
### Control Chart Analysis: Mean

## Conclusion (3 minutes)
