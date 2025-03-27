# MKT3434_2025 - Machine Learning Course GUI

This project is a GUI application developed for the **MKT3434 Machine Learning Course**.  
It allows users to apply machine learning models, visualize results, and explore data preprocessing methods.

---

### 1. 📊 Loss Function Selection
Each model includes appropriate **loss function options**:

- **Regression Models (SVR, Linear Regression):**
  - MSE
  - MAE
  - Huber Loss

- **Classification Models (SVM, Logistic Regression):**
  - Cross-Entropy Loss
  - Hinge Loss

> Selected loss is shown in the **status bar** after training.

---

### 2. 🔧 SVR & SVM Kernel and Parameters
- Models now support **kernel selection**: `linear`, `rbf`, `poly`
- SVR includes adjustable `C` and `epsilon` parameters

---

### 3. 📂 Missing Data Handling
User can select **how to handle missing values** before training:
- No Handling
- Mean Imputation
- Interpolation
- Forward Fill
- Backward Fill

> Selected method is applied before train/test split and shown in the status bar.

---

### 4. 📈 Naive Bayes with Custom Priors
- **var_smoothing** value is customizable
- Users can enter **prior probabilities**, e.g., `[0.5, 0.5]`

---

## 💡 How to Use

1. Select a dataset or load a CSV file.
2. Choose test split ratio, missing data handling, and scaling.
3. In the **Classical ML tab**, pick a model and adjust parameters.
4. Select a **loss function**.
5. Click **Train Model**.
6. Check results in the **visualization panel** and **metrics box**.


## 👨‍🎓 Student Info

- **Name**: Safa Emre Beytekin  
- **Student ID**: 2206A020 
- **Course**: MKT3434  
- **University**: Yildiz Technical University

