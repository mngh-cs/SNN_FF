# SNN_FF
Training Spiking Neural Network with the Forward-Forward Algorithm

Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm

This repository contains the implementation and experiments from the paper:

Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm
Mohammadnavid Ghader, Saeed Reza Kheradpisheh, Bahar Farahani, Mahmood Fazlali (2025)

📌 Overview

Spiking Neural Networks (SNNs) provide a biologically inspired way of computation by mimicking the brain’s spike-based communication. However, training them with traditional backpropagation (BP) is inefficient and biologically implausible.

This work introduces a training framework for SNNs based on the Forward-Forward (FF) algorithm—a method proposed by Geoffrey Hinton that replaces the forward–backward training cycle with two forward passes.

✨ Key Features

No backpropagation: training is forward-only.

Layer-wise localized learning: each layer learns independently using a “goodness” measure.

Biological plausibility: avoids issues like weight transport and global error signals.

Neuromorphic hardware-friendly: efficient and well-suited for low-power implementations.

Competitive results: accuracy comparable to or better than backpropagation-trained SNNs.

🧪 Datasets

The method was tested on both static and spiking/temporal datasets:

Static: MNIST, Fashion-MNIST, Kuzushiji-MNIST, CIFAR-10

Spiking: N-MNIST, SHD (Spiking Heidelberg Digits)

📊 Results

Our Forward-Forward-trained SNN:

Achieves 98.34% accuracy on MNIST with a lightweight architecture.

Outperforms other FF-based SNNs on static datasets.

Performs competitively with state-of-the-art BP-based SNNs on spiking datasets like SHD.

Requires fewer time steps (10), reducing computational cost compared to traditional SNN training.

⚙️ Implementation

Framework: PyTorch with snntorch

Training setup:

Optimizer: Adam

Learning rate: 0.001

Epochs: 300 (500 for SHD)

Batch size: 4096

Neuron model: Leaky Integrate-and-Fire (LIF) with learnable membrane time constants

🚀 How It Works

Input samples are combined with labels to generate positive (true label) and negative (wrong label) pairs.

Two forward passes are performed:

Positive pass → maximize neuron “goodness”

Negative pass → minimize neuron “goodness”

Each layer updates weights locally based on a contrastive loss.

During inference, labels are tested by embedding and selecting the one with the highest goodness score.

📂 Repository Structure
├── data/              # Datasets (or links/instructions to download)
├── models/            # Network definitions
├── experiments/       # Training scripts & configs
├── results/           # Logs, plots, and accuracy tables
├── README.md          # Project overview (this file)

🔮 Future Work

Optimization for neuromorphic chips.

Extension to larger and more complex datasets.

Exploration of hybrid learning methods combining FF with other biologically inspired algorithms.

📄 Citation

If you use this code, please cite:

@article{ghader2025ff-snn,
  title={Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm},
  author={Ghader, Mohammadnavid and Kheradpisheh, Saeed Reza and Farahani, Bahar and Fazlali, Mahmood},
  year={2025}
}
