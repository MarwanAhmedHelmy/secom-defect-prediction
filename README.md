SECOM Manufacturing Defect Prediction

Predicting semiconductor manufacturing defects from noisy, high-dimensional sensor data — optimizing for catching failures, not raw accuracy.

The problem

On a semiconductor manufacturing line, hundreds of sensors record process conditions for every unit that comes off the line. Only a small fraction of units actually turn out defective — in this dataset, just 6.6%. That imbalance is the real challenge: a model that simply predicts "pass" for everything would already be right 93% of the time, while catching zero real defects. The goal wasn't raw accuracy — it was building a model that actually catches failing units before they leave the line.

The data

The SECOM dataset (UCI Machine Learning Repository) contains 1,567 production records, each described by 590 anonymized sensor and process-measurement features, with a binary pass/fail label (1,463 pass / 104 fail — 6.6% failure rate).

Raw dataset files (secom.data, secom_labels.data, secom.names) are included in this repo as a fallback. By default, the notebook downloads the same dataset automatically via kagglehub (paresh2047/uci-semcom); if Kaggle access isn't available, load these files directly instead.

Pipeline
Cleaning & feature filtering — Imputed missing sensor readings, dropped near-constant (zero-variance) features, and reduced redundancy among highly correlated sensors.
Exploratory data analysis — Examined sensor distributions and correlation with the target to separate signal from noise.
Stratified train/test split — Preserved the 6.6% failure ratio in both sets so evaluation reflected the real class balance.
Imbalance handling — Compared class-weighting and SMOTE oversampling (applied to the training data only, to avoid leaking synthetic samples into evaluation).
Model comparison — Trained and compared Logistic Regression, Random Forest, and SVM.
Imbalance-aware evaluation — Scored every model on Precision, Recall, F1 and ROC-AUC rather than accuracy, since recall on the defect class was what actually mattered.
Threshold tuning — Adjusted the decision threshold after training to push defect-detection recall higher, at a deliberate, measured cost to precision.
Results
Model	Method	Precision (Fail)	Recall (Fail)	F1 (Fail)	ROC-AUC
Logistic Regression	class_weight (threshold=0.5)	0.105	0.190	0.136	0.621
Random Forest	SMOTE (threshold=0.2)	0.162	0.762	0.267	0.780
SVM	SMOTE (threshold=0.5)	0.500	0.048	0.087	0.654

The best configuration — Random Forest with SMOTE and a lowered decision threshold — catches 76% of defective units on the test set. On a production line, that's the number that matters: far fewer failing units slip through undetected, at a deliberate, measured cost in precision.

Tech stack

Python · Pandas / NumPy · Scikit-learn · imbalanced-learn (SMOTE) · Matplotlib / Seaborn

Repository structure
secom-defect-prediction/
├── secom_defect_prediction.ipynb   # full notebook: EDA, cleaning, modeling, evaluation
├── secom.data                      # raw sensor readings (fallback dataset)
├── secom_labels.data               # pass/fail labels (fallback dataset)
├── secom.names                     # dataset field descriptions
└── README.md
Running it
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn kagglehub
Open secom_defect_prediction.ipynb in Jupyter.
The notebook downloads the dataset via kagglehub by default. If Kaggle access isn't configured, load secom.data, secom_labels.data and secom.names directly from the repo instead.

Part of Marwan Ahmed's Data & ML portfolio
