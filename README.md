# Federated-Learning-for-chest-X-ray-classification
# Research Question
How does data heterogeneity affect the performance of Federated Learning for chest X-ray classification?
# Motivation
Federated learning (FL) is a machine learning approach that enables the training of a shared AI model using data from numerous decentralized edge devices or servers. This process occurs without the need to exchange the local data samples
Heterogeneity of data and devices: Clients in a federated learning network can vary significantly in terms of their data distribution (non-independent and identically distributed, or non-IID data) and their computational capabilities (device hardware, network connectivity). This diversity can impact model convergence and overall performance.

# Hypothesis
As client data become increasingly heterogeneous, the performance of standard FedAvg will decrease compared with centralized training.

# dataset
NIH Chest X-rays
Over 112,000 Chest X-ray images from more than 30,000 unique patients
