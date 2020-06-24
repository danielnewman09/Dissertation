# Presentation Outline

## Introduction: (7 minutes)

## Slide 1: Title (0:15)

Good afternoon and thank you for coming to my thesis defense. Before I begin, I want to draw your attention to the links at the bottom of this slide. I have made these slides, and the code used to generate them available online. All of the data used for this research is also available on Github.

### Slide 2: The Past Decade (2:00)

With that said, I want to begin by providing some context for this research. The work I'm about to present would have been impossible ten, five, or for some elements even two years ago. The advancements in embedded computing, machine learning, and Internet of Things have dramatically changed the technological landscape in the past decade. Many of these advancements are built on technologies first introduced in the mid 2000's. A couple of examples, shown here, are the 32-bit microprocessor architecture and the Numpy library. 
  
### Slide 3: Research Questions (1:15)

This research is specifically interested in how recent technological changes can be leveraged in a machine monitoring application. To that end, I am interested in answering these questions. 

...

Again, these questions essentially boil down to investigating how these new embedded computing and internet of things tools can be leveraged in a manufacturing environment

### Slide 4: Thesis Contributions (2:30)

Through investigating these questions, I will present an integrated Internet of Things architecture and data acquisition strategy that facilitates manufacturing machine health monitoring. Specifically, this architecture incorporates three major contributions. First, ...

...While previous work has addressed individual aspects of the work, a critical advancement from this thesis is the integration of all these elements from data acquisition to health monitoring. As a result, this work shows a truly end-to-end system which can be scaled to a manufacturing environment.

### Slide 8: Demonstration (1:00)

All of these elements are demonstrated at a high-level in this video of a simple end milling process.

### Slide 9: Outline

## Background (5 Minutes)

### Slide 10: Industrial Internet of Things

The industrial internet of things is an important concept in improving manufacturing process efficiency, rapidly responding to changes, and preventing unscheduled downtime. This figure shows a simple heriarchy of the main components involved in this .

...

For this presentation, I'll be going into more depth specifically on the data acquisition and message transmission layers of this framework. I want to particularly focus on the means by which CNC controller data and external sensor data are extracted and processed.

### Slide 11: CNC Controller Protocols

For modern CNC machines, their controllers often come equipped with software that allows important data to be readily extracted from them.

### Slide 12: Sensor Data Acquisition

In addition to controller data, external sensors can be used to enrich the data from the machine. Using sensors such as thermocouples, accelerometers, and current sensors, highly detailed information can be captured from these devices. 

### Slide 13: Health Monitoring

Health monitoring is an important application for these controller and sensor data streams. 

### Slide 15: Spectral Power Approximation


## Digital Architecture for Health Monitoring (6 minutes) [0:25]

Now I'll get into the first contribution of this thesis, the integrated digital architecture. This section focuses on acquiring and parsing the types of data that are necessary for manufacturing health monitoring. Specifically, this strategy seeks to record sensor and controller data to paint a complete picture of machine health and utilization.

### Slide 19: MQTT-Based Framework [1:30]

Now, when transmitting data in an IoT infrastructure, it's important to begin with an efficient and effective messaging protocol. In this case, MQTT is an excellent candidate. This is a publish-subscribe protocol which has low overhead and uses a simple topic syntax which makes subscribing to a desired data stream extremely easy. As this figure shows, ...


### Slide 20: MQTT Message Definitions 

To use this protocol in a large manufacturing environment, it's important to establish a data structure. For MQTT, this includes the topic and payload as shown in this slide.

### Slide 21: Controller/Sensor Payload Example

With this message structure established, here are two example messages - one from a CNC controller and one from an accelerometer.

### Slide 23: Contextual Data Acquisition and Training



### Slide 26: Emco Warm-Up Program
### Slide 27: Emco Warm-Up History

## Edge-Deployable ML Tools (6 minutes)



### Open Source Software
### Example Dataset
### Feature Normalization
### Deployed Models
### Model Deployment
### Full Latency Comparison: Classifiers

## Integrated Edge Device (8 minutes)

### Open Source Components
### CNC Controller Integration
### Analog Data Acquisition
### Edge Device Diagram
### Device Benchmarking
### Local Compute Latency
### Cloud Compute Latency Comparison

## Case Study (10 minutes)

### Experimental Setup
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
