# Histopathologic Cancer Detection Ensemble

Hey there! 👋

This project tackles cancer detection in histopathology images using a powerful ensemble approach — combining the best of deep learning and classic machine learning to get reliable, high-performance results.

---

## What’s inside?

- **6 different CNNs and Transformer models** trained separately on histopathology images.  
- **Ensembling of these base models** to capture diverse patterns and features.  
- **3 different meta learners** (Logistic Regression, CatBoost, XGBoost) trained on the base models’ stacked predictions.  
- **Meta-learner ensemble** — combining those meta models via weighted soft voting for even better accuracy.  
- **Threshold tuning** to find the perfect balance between catching as many cancers as possible (high recall) and keeping false alarms low (high precision).  
- **Thorough evaluation** using precision, recall, F1 score, confusion matrices, and classification reports.

---

## Why this matters?

Cancer detection from histopathology images is challenging but crucial. Early, accurate identification can save lives. This project blends multiple neural architectures and classic ML methods, stacking their strengths to create a reliable diagnostic aid.

---

## Results you can trust

- Recall ~0.99 — very few missed cancer cases.  
- Precision >0.96 — keeping false positives low.  
- F1 scores consistently near 0.98 — balancing precision and recall for clinical usefulness.

---

## What’s next?

- Create pesudo-labeled data from unlabled data.
- Try level-2 stacking to push performance further.    
- Wrap everything in an API or app for easy use by medical professionals.

---

## Let’s connect!

Questions, suggestions, or just want to chat about medical AI? Open an issue or get in touch — I’m all ears!

---

Thanks for stopping by! Happy coding and here’s to making a difference! 

--- 

Colab Notebook Link : https://colab.research.google.com/drive/1j3sN92REYKObW8JKnhQrTc5-Pss2I30y?usp=sharing
