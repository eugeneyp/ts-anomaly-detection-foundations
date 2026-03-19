# SWaT Dataset: Secure Water Treatment System

Downloaded on March 17, 2026 from https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system

This dataset contains sensor and actuator measurements collected from the Secure Water Treatment (SWaT) testbed. It includes both normal operational data and cyber-attack scenarios, simulating real-world industrial control system intrusions. The dataset is suitable for research in anomaly detection, intrusion detection, cybersecurity, and machine learning applications in critical infrastructure.

The SWaT dataset is a benchmark dataset widely used in industrial control system (ICS) security research. It consists of time-series sensor and actuator data collected from a real-world water treatment testbed. It includes both normal and attack scenarios, making it highly suitable for tasks such as anomaly detection, intrusion detection, time-series classification, and ICS fault detection. ** Data Overview**

The dataset contains timestamped measurements from various sensors and actuators across multiple stages of the water treatment process. Key columns include:

Timestamp — Date and time of the recorded data point

FIT101 — Flow Indicator Transmitter at stage 1

LIT101 — Level Indicator Transmitter at stage 1

MV101 — Motorized Valve at stage 1

P101, P102 — Pumps at stage 1

AIT201, AIT202, AIT203 — Analyzer Indicators for pH, conductivity, and ORP at stage 2

FIT201 — Flow Indicator Transmitter at stage 2

MV201 — Motorized Valve at stage 2

P201, P202, P203, P204, P205, P206 — Pumps at stage 2

DPIT301 — Differential Pressure Indicator Transmitter at stage 3

FIT301, LIT301 — Flow and Level indicators at stage 3

MV301, MV302, MV303, MV304 — Motorized Valves at stage 3

P301, P302 — Pumps at stage 3

AIT401, AIT402 — Analyzer Indicators at stage 4

FIT401, LIT401 — Flow and Level indicators at stage 4

P401, P402, P403, P404 — Pumps at stage 4

UV401 — UV disinfection unit

AIT501, AIT502, AIT503, AIT504 — Analyzer Indicators at stage 5

FIT501, FIT502, FIT503, FIT504 — Flow indicators at stage 5

P501, P502 — Pumps at stage 5

PIT501, PIT502, PIT503 — Pressure Indicator Transmitters at stage 5

FIT601 — Flow Indicator Transmitter at stage 6

P601, P602, P603 — Pumps at stage 6

Normal/Attack — Label indicating whether the data point corresponds to normal or attack operation

**Files Included**

normal.csv — Normal operational data

attack.csv — Cyber attack data

merged.csv — Combined data (normal + attack)

**Use Cases**

Anomaly detection in industrial control systems

Binary or multiclass classification for attack type detection

Real-time monitoring systems for critical infrastructure

Time-series forecasting under adversarial conditions

## Files
### normal.csv
This file contains normal operational data collected from the Secure Water Treatment (SWaT) testbed during stable and attack-free conditions. All sensor and actuator values represent the standard functioning of the industrial control system without any external interference.

The data reflects the baseline behavior of the physical and cyber components in the water treatment process, serving as a ground truth reference for model training and evaluation in anomaly detection and cybersecurity research. Researchers can use this dataset to learn normal temporal patterns, system dynamics, and correlations among process variables across different treatment stages.

### attack.csv
This file contains data recorded during controlled cyber-attack experiments on the Secure Water Treatment (SWaT) testbed. The attacks were intentionally introduced to simulate malicious intrusions targeting various components such as sensors, actuators, pumps, and valves.

Each data point is labeled as either normal or attack, enabling researchers to develop and evaluate models for intrusion detection, anomaly classification, and fault tolerance in industrial control systems. The inclusion of both normal and compromised states makes this dataset valuable for studying the impact of cyber-physical attacks and developing resilient monitoring systems.

### merged.csv
This file merges both normal and attack data from the SWaT testbed into a single, continuous dataset. It provides a realistic representation of an operational industrial environment where normal operation transitions into abnormal (attack) conditions over time.

The combined dataset is particularly useful for supervised learning, binary classification (normal vs. attack), and time-series anomaly detection tasks. It helps in evaluating how models perform under mixed scenarios, making it suitable for developing real-time monitoring and intrusion detection systems in critical infrastructure.