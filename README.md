# Malware Classification Project

This project focuses on **malware classification** using deep learning (CNNs) and a **Random Forest** for comparison. The dataset consists of **hexadecimal `.bytes` files**, which are transformed into images (PNG format) to leverage computer vision techniques. We developed **two distinct classifiers**:

1. **Multiclass Classifier**  
   - **Objective**: Identify the exact **malware family** among **25 known malware classes** (e.g., Allaple.A, Allaple.L, etc.).  
   - **Example**: Predict whether an input sample belongs to one of the 25 malicious categories.

2. **Binary Classifier**  
   - **Objective**: Distinguish **malware** from **benign** software.  
   - **Example**: Given an input sample, classify it as either malicious or safe (non-malware).

---

## 1. Data Conversion

- **Hex-to-Image**:  
  1. Read each `.bytes` file (hex dump).  
  2. Parse the hexadecimal values (`3A`, `4F`, `??` → missing is zero).  
  3. Reshape the resulting matrix into a 2D array.  
  4. Convert the array to a grayscale image (PNG).

- **Result**: A collection of PNG images where each pixel’s intensity corresponds to the numeric value of the original hex data.

---

## 2. Data Organization and Augmentation

- **Directory Structure**:  
  - Each of the **25 malware families** and **benign files** has its own directory, making **26 total categories** in the dataset (for multiclass).  
  - For binary classification, files are labeled simply as **malware** or **benign**.

- **ImageDataGenerator**:
  - Splits data into **train** and **test** subsets (e.g., 70% / 30%).  
  - Can perform **real-time data augmentation** (random rotations, flips, shifts) to improve generalization.

---

## 3. Deep Learning Approach (CNN)

### 3.1 CNN Architecture

- **Convolutional Blocks**:  
  1. **Conv2D** → **BatchNormalization** → **MaxPooling2D** → **Dropout**  
  2. **Conv2D** → **BatchNormalization** → **MaxPooling2D** → **Dropout**  
  3. **Conv2D** → **BatchNormalization** → **MaxPooling2D** → **Dropout**

- **Final Layers**:  
  - **Flatten**  
  - **Dense(256, activation='relu')** with Dropout  
  - **Output** layer:
    - **Multiclass**: 25 (or 26) neurons (depending on whether benign is included), with `softmax`.
    - **Binary**: 1 neuron with `sigmoid` (if purely malware vs. benign).

- **Compilation**:
  - **Optimizer**: Adam  
  - **Loss**: 
    - `categorical_crossentropy` for the **multiclass** setup.  
    - `binary_crossentropy` for the **binary** classifier.  
  - **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC, etc.

### 3.2 Training Procedure
- **k-Fold Cross-Validation**:  
  - Data is split into k folds (e.g., 5).  
  - Train on k-1 folds, validate on the remaining fold, repeat k times.  
  - This robustly evaluates model performance while mitigating overfitting.

- **Handling Class Imbalance**:  
  - **Class weights** ensure minority classes have sufficient influence on the loss function.

---

## 4. Random Forest Classifier

- **Purpose**: Compare a traditional machine learning approach with the CNN method.  
- **Feature Extraction**:  
  - Either flatten the PNG images into feature vectors or compute domain-specific features.  
- **Training**:  
  - Train a **Random Forest** with a suitable number of trees (e.g., 100–500).  
  - **Multiclass** scenario: 25–26 classes.  
  - **Binary** scenario: malware vs. benign.

- **Comparison**:  
  - Evaluate both CNN and Random Forest on the same folds.  
  - Compare metrics (accuracy, precision, recall, F1, AUC) and discuss pros/cons:
    - **CNN** is powerful for image-based features but can require more computational resources.  
    - **Random Forest** is simpler, often faster to train, but may require careful feature engineering for image data.

---

## 5. Two Classifiers in Detail

1. **Multiclass Classifier**  
   - **Goal**: Identify **which malware family** the sample belongs to (25 families + possibly 1 benign category).  
   - **Use Cases**: Security researchers or antivirus software that must detect the specific malware strain.

2. **Binary Classifier**  
   - **Goal**: Distinguish **malware vs. benign**.  
   - **Use Cases**: Quick triaging to see if a file is safe or malicious.

In practice, **both** classifiers can be useful: a **binary** check might be a fast filter for general malware detection, while the **multiclass** model provides deeper insights into malware strain types.

---

## 6. Model Evaluation

- **Confusion Matrix**: Shows True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) for each class (or for the binary scenario).
- **Precision & Recall**:  
  - **Precision** = TP / (TP + FP)  
  - **Recall** = TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC Curve & AUC**: Evaluates ranking performance; higher AUC implies better discriminative ability.
- **Accuracy**: Overall correctness of classification (can be misleading in imbalanced datasets, hence the need for multiple metrics).

**Example**:  
- In multiclass classification, each family is evaluated separately in the confusion matrix.  
- In the **binary** classifier, the matrix simply has “Malware” vs. “Benign” cells.

---

## 7. Collaboration

This project was carried out in **collaboration with Sanda Dhouib**. Together, we designed and implemented the data processing pipelines, the CNN-based architecture, and the Random Forest comparison, ensuring a comprehensive approach to malware detection and classification.

---

## 8. Conclusion

By converting `.bytes` files into images and using **CNNs**, we leverage visual patterns for **malware detection**. Training two separate classifiers (one **multiclass**, one **binary**) broadens our approach:

- **Multiclass**: Pinpoints the exact family among 25 malware types.  
- **Binary**: Quickly flags malicious vs. benign files.

Comparing the **CNN** with a **Random Forest** reveals trade-offs in **training time, feature engineering, and predictive performance**. This comprehensive evaluation—via **k-fold cross-validation**, **class weighting**, **precision/recall/F1**, and **AUC**—helps ensure reliable detection accuracy in real-world cybersecurity scenarios.

---

## References

```latex
\begin{thebibliography}{99}

\bibitem{CNN for Image Classification}
A. Krizhevsky, I. Sutskever and G. E. Hinton, ``ImageNet Classification with Deep Convolutional Neural Networks,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}.
\url{https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf}

\bibitem{CNNs for Malware Detection}
Gibert, D., Mateu, C. and Planes, J. \emph{The rise of machine learning for detection and classification of malware: Research developments, trends, and challenges.}
\url{https://www.sciencedirect.com/science/article/pii/S1084804519303868}

\bibitem{CNN Explanation}
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. \emph{Gradient-based learning applied to document recognition.}\\
\url{https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939}

\bibitem{CNN Concepts}
Towards Data Science: Introduction to CNNs. \\
\url{http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf}

\bibitem{CNN Concepts}
Songqing Yue, Tianyang Wang, Imbalanced Malware Images Classification: a CNN based Approach.\\
\url{https://arxiv.org/abs/1708.08042}

\bibitem{CNN Concepts}
M. Kalash, M. Rochan, N. Mohammed, N. D. B. Bruce, Y. Wang and F. Iqbal, \emph{"Malware Classification with Deep Convolutional
Neural Networks,"} IFIP International Conference on New Technologies, Mobility and Security (NTMS).\\
\url{https://ieeexplore.ieee.org/document/8328749}

\end{thebibliography}
