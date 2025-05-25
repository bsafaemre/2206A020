# MKT3434_2025 - Machine Learning Course GUI

This project is a GUI application developed for the **MKT3434 Machine Learning Course**.  
It enables users to apply machine learning models, visualize results, and explore preprocessing and dimensionality reduction techniques.

---

## 🧠 Key Features

### 1. 📊 Loss Function Selection
Each model includes appropriate **loss function options**:

- **Regression Models (SVR, Linear Regression):**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Huber Loss

- **Classification Models (SVM, Logistic Regression):**
  - Cross-Entropy Loss
  - Hinge Loss

> Selected loss is displayed in the **status bar** after training.

---

### 2. 🔧 SVR & SVM Kernel and Parameters
- Kernel options: `linear`, `rbf`, `poly`
- Adjustable parameters: `C`, `epsilon` (for SVR), `degree` (for SVM)

---

### 3. 📂 Missing Data Handling
User can select how to handle missing values before training:
- No Handling
- Mean Imputation
- Interpolation
- Forward Fill
- Backward Fill

> Method is applied before train/test split and reflected in the status bar.

---

### 4. 📐 Dimensionality Reduction (New Tab Added)
New tab includes support for common dimensionality reduction techniques:

| Method | Notes |
|--------|-------|
| **PCA** | User-selectable `n_components`, visualizes **cumulative explained variance** |
| **LDA** | Supervised projection with class separation, accuracy score shown |
| **K-Means** | `k` selection via Elbow method, **silhouette score** calculated |
| **t-SNE** | 2D/3D projection with **perplexity input**, uses **Plotly if selected** |
| **UMAP** | 2D/3D interactive projection, supports `n_neighbors` and `n_components` |

> **Plotly integration** is used for **t-SNE and UMAP**, with support for 2D/3D switching depending on `n_components`.

---

### 5. 📦 Feature Engineering and CV Options
- **Scaling options**: Standard, MinMax, Robust
- **Custom train/test splits**: 80/20, 70/15/15, 60/20/20
- **k-Fold Cross-Validation**:
  - User selects number of folds (`k`)
  - Supported metrics: Accuracy, MSE, MAE
  - Mean ± std.dev reported in **metrics box**

---

## 🖼️ Visualization
- **matplotlib** is used for static plots (e.g., loss curves, PCA variance, clustering).
- **plotly** is optionally used for interactive 2D/3D visualizations of t-SNE & UMAP.
- Visualization panel automatically updates after training or projection.

---

---

## 🤖 Deep Learning Module (NEW)

A new **"Deep Learning"** tab has been added to the GUI. Users can now build and train deep neural networks without writing code.

- **Supported Architectures**:
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (LSTM, GRU)
  - Pretrained Models: VGG16, ResNet50, MobileNetV2

- **Custom Layer Configuration**:
  - Add layers dynamically (Dense, Conv2D, Dropout, MaxPooling2D, Flatten, LSTM, GRU)
  - Support for activation functions, kernel sizes, L2 regularization, and return_sequences (for RNNs)

- **Training Options**:
  - Optimizer selection: Adam, SGD, RMSprop
  - Learning rate scheduler: Step Decay, Exponential Decay
  - EarlyStopping support with patience value
  - Image augmentation: Rotation, Flip, Zoom, Brightness
  - Save/Load Keras `.h5` model files

---

## 🎮 GAN Training Support (NEW)

A simple **Generative Adversarial Network (GAN)** module is added under the "GAN" tab.

- Generator and Discriminator are built-in
- Dataset: MNIST
- Adjustable `epochs` and `batch size`
- Shows generated sample image every 10 epochs
- Logs GAN training progress live

---

## 📋 Real-Time Training Logs (NEW)

- A **live log panel** has been added next to the canvas.
- During CNN, RNN, or GAN training, real-time feedback is shown (loss, accuracy, gradients, etc.).
- Helps track training progress without using the console.

---

## ⚙️ Optimizer and Regularization Enhancements (NEW)

New configurations added to improve training control:

- Optimizer selection:
  - Adam
  - SGD
  - RMSprop

- Regularization support:
  - L2 kernel regularization for Dense, Conv2D, LSTM, GRU
  - Dropout layers (with adjustable rate)



## 👨‍🎓 Student Info

- **Name**: Safa Emre Beytekin  
- **Student ID**: 2206A020 
- **Course**: MKT3434  
- **University**: Yildiz Technical University

---

## 🔧 Setup Instructions

### ✅ Requirements
Make sure the following packages are installed:

```bash
pip install numpy pandas matplotlib PyQt6 scikit-learn tensorflow torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia umap-learn plotly

