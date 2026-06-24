# Democratizing Cardiovascular Diagnostics: Overcoming the "Small Data Gap" in Deep Learning

## Project Overview
Cardiovascular disease (CVD) is the leading cause of death globally, with nearly 650 million people living with the condition[cite: 1]. The objective of this project is to determine if standard, easily obtainable clinical data can be used to precisely predict heart failure using machine learning. 

Specifically, this study systematically evaluates whether deeply regularized Deep Learning (TensorFlow) architectures can overcome overfitting to provide a diagnostic advantage over traditional Machine Learning algorithms (Scikit-Learn) when applied to small-scale, tabular clinical datasets.

## Dataset & Preprocessing
The project utilizes a clinical dataset containing 918 patient records with 11 distinct clinical features. To ensure clinical validity, rigorous data preprocessing was applied:
* **Handling Biological Anomalies:** 172 patient records contained a biologically impossible cholesterol level of 0 mm/dl. To preserve 20% of the dataset, these were rescued using median imputation rather than deletion[cite: 1].
* **Categorical Encoding:** Text-based categorical features (e.g., Sex, Chest Pain Type) were transformed into binary matrices using One-Hot Encoding.
* **Feature Scaling:** A `StandardScaler` was applied to normalize physiological measurements, ensuring no single feature mathematically dominated the algorithms.

## Experimental Design
In medical diagnostics, the cost of a False Negative (sending a sick patient home) is catastrophic. Therefore, all models were optimized specifically to maximize **Recall (Sensitivity)** to ensure no sick patient was left undetected. A systematic 7-experiment pipeline was executed:

| Exp # | Design Adjustment | Val Recall | F1-Score | Clinical Result |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Scikit-Learn Baseline (Random Forest)** | 0.8824 | 0.8867 | Strong baseline, but missed 12 actively sick patients[cite: 1]. |
| **2** | Naive Deep TF Baseline (Unregularized) | 0.8725 | 0.8683 | Severely overfit the small dataset; missed 13 patients[cite: 1]. |
| **3** | Lower Learning Rate (Adam 0.0001) | 0.8922 | 0.8966 | Prevented rapid conclusions; improved clinical sensitivity[cite: 1]. |
| **4** | Functional API (Wide & Shallow 128) | 0.8824 | 0.8867 | Stable architecture, but failed to beat the low learning rate[cite: 1]. |
| **5** | Uniform Dropout Regularization (0.3) | 0.8824 | 0.8696 | Built redundant diagnostic pathways[cite: 1]. |
| **6** | L2 Weight Decay Penalty (0.01) | 0.8922 | 0.8922 | Continuous mathematical tax successfully raised Recall[cite: 1]. |
| **7** | **Early Stopping + Dropout + Low LR** | **0.9216** | **0.8952** | **Ultimate Model: Missed only 8 sick patients**[cite: 1]. |

## Final Results
The final optimized deep learning architecture (Experiment 7) successfully surpassed the traditional machine learning baseline. By combining a reduced gradient velocity, forced structural redundancy, and dynamic halting (Early Stopping), the model achieved a **peak Recall of 92.16%**. 

This project proves that systematically regularized neural networks can serve as a highly sensitive, accessible, and life-saving diagnostic tool, successfully identifying 94 sick patients and reducing fatal false negatives to just 8 cases in the testing population.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deep Learning:** TensorFlow / Keras
* **Data Visualization:** Matplotlib, Seaborn