# Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm

## 📌 Overview

Spiking Neural Networks (SNNs) provide a biologically inspired way of computation by mimicking the brain’s spike-based communication. However, training them with traditional **backpropagation (BP)** is inefficient and biologically implausible.

This work introduces a training framework for SNNs based on the **Forward-Forward (FF) algorithm**—a method proposed by Geoffrey Hinton that replaces the forward–backward training cycle with **two forward passes**.

### ✨ Key Features

* **No backpropagation:** training is forward-only
* **Layer-wise localized learning:** each layer learns independently using a “goodness” measure
* **Biological plausibility:** avoids issues like weight transport and global error signals
* **Neuromorphic hardware-friendly:** efficient and well-suited for low-power implementations
* **Competitive results:** accuracy comparable to or better than backpropagation-trained SNNs

* ANN_FF Folder: The codes related to comparing our spiking method with other existing ANN models
* SNN_FF Folder: The codes related to comparing our spiking method with other existing SNN models

---

## 🧪 Datasets

The method was tested on both **static** and **spiking/temporal** datasets:

* **Static:** MNIST, Fashion-MNIST, Kuzushiji-MNIST, CIFAR-10
* **Spiking:** N-MNIST, SHD (Spiking Heidelberg Digits)

---

## 📊 Results

Our Forward-Forward-trained SNN:

* Outperforms other FF-based SNNs on static datasets
* Performs **competitively with state-of-the-art BP-based SNNs** on spiking datasets like SHD
* Requires **fewer time steps (10)**, reducing computational cost compared to traditional SNN training

---

## 📄 Citation

If you use this code, please cite:

```
@article{Ghader2025BackpropagationfreeSN,
  title={Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm},
  author={Mohammadnavid Ghader and Saeed Reza Kheradpisheh and Bahar Farahani and Mahmood Fazlali},
  journal={ArXiv},
  year={2025},
  volume={abs/2502.20411},
  url={https://api.semanticscholar.org/CorpusID:276725450}
}
```
