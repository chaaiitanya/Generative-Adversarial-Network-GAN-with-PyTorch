# Generative-Adversarial-Network-GAN-with-PyTorch

# Generative Adversarial Network (GAN) with PyTorch

This repository contains a step-by-step implementation of a **Generative Adversarial Network (GAN)** using PyTorch. The project focuses on generating realistic handwritten digits inspired by the MNIST dataset. It includes detailed code for data preprocessing, model creation, training, and visualization of results.

---

## **Features**
- **Dataset**: Utilizes the MNIST dataset with transformations for augmentation and normalization.
- **Discriminator Network**: A convolutional neural network that classifies images as real or fake.
- **Generator Network**: A transposed convolutional neural network that generates realistic images from random noise.
- **Loss Functions**: Binary Cross Entropy (BCE) loss for real and fake image classification.
- **Training Loop**: Alternating updates for the discriminator and generator to achieve adversarial training.
- **Visualization**: Displays generated images at different training epochs to show progress.

---

## **Key Highlights**
1. **Configurations**:
   - Noise dimension: 64
   - Batch size: 128
   - Learning rate: 0.0002
   - Optimizer: Adam with `beta_1=0.5` and `beta_2=0.99`
   - Training epochs: 20

2. **Discriminator Architecture**:
   - Convolutional layers with increasing filters (16 → 32 → 64).
   - Batch normalization and LeakyReLU activation.
   - Fully connected layer for binary classification.

3. **Generator Architecture**:
   - Transposed convolutional layers with decreasing filters (256 → 128 → 64 → 1).
   - Batch normalization and ReLU activation (Tanh for output layer).
   - Generates 28x28 grayscale images from random noise.

4. **Training Process**:
   - Discriminator is trained to distinguish real from fake images.
   - Generator is trained to fool the discriminator by producing realistic images.
   - Loss values for both networks are logged and plotted.

5. **Results**:
   - Generated images improve in quality over epochs.
   - Final generator can produce realistic handwritten digits.

---

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/gan-pytorch.git
   cd gan-pytorch
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib tqdm
   ```

3. Run the Jupyter Notebook or Python script to train the GAN:
   ```bash
   jupyter notebook Generative-Adversarial-Network.ipynb
   ```

4. Visualize generated images during or after training.

---


