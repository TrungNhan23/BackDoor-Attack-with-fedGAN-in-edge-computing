# Backdoor Attack in Federated Learning

## 1. Introduction

Federated Learning (FL) is a decentralized machine learning paradigm that enables multiple clients to collaboratively train a model without sharing raw data. However, FL is vulnerable to various security threats, one of which is the Backdoor Attack. In a backdoor attack, an adversary attempts to implant hidden malicious behavior into a global model without affecting its performance on clean data.

This document presents a study and implementation of backdoor attacks in a Federated Learning setup.

## 2. Objective

The main objectives of this project are:

> To implement a backdoor attack in an FL setting.

> To analyze and compare clean-label and dirty-label attack methods.

> To evaluate the attack effectiveness using metrics like Clean Accuracy (CA) and Attack Success Rate (ASR).

## 3. Attack Types (Backdoor attack)

### 3.1 Clean-label Attack

> Definition: The attacker uses poisoned inputs but keeps the original label.

> Example: Add a trigger to images of digit '1' but keep the label as '1'.

> Goal: The model learns the association between the trigger and misclassifies future inputs containing the trigger.

### 3.2 Dirty-label Attack

> Definition: The attacker injects poisoned data and assigns incorrect labels (target class).

> Example: Generate digit '1' with adversarial noise and label it as '7'.
 
> Goal: The model classifies any input with similar noise as class '7'.

## 4. Methodology

### 4.1 Dataset

> MNIST dataset is used for both training and testing.

> Poisoned Data is generated using adversarial techniques (e.g., FGSM).

### 4.2 Federated Learning Setup

> Multiple clients train locally on clean or poisoned data.

> A global server aggregates updates using Federated Averaging (FedAvg).

> Some clients act as adversaries with poisoned data.

### 4.3 GAN Generation (DCGAN)

> A GAN is used to generate fake digit '1' images.

> A trigger (adversarial noise) is added to these images.

> In dirty-label attack, the images are relabeled as '7'.

## 5. Evaluation Metrics

### 5.1 Clean Accuracy (CA)

> Accuracy of the model on clean (non-poisoned) test data.

> Indicates overall performance.

### 5.2 Attack Success Rate (ASR)

> Percentage of poisoned test samples that are classified as the target label.

> Measures the strength of the backdoor.

## 6. Results Experiments


