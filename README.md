# 🧩 Text Classification with Active Learning

This project implements a **Multinomial Naive Bayes text classifier** enhanced with **entropy-based Active Learning** to reduce labeling effort while maintaining high accuracy.  
The model is trained on a **Sentiment140-derived binary sentiment dataset**, demonstrating efficient learning with limited labeled data.

---

## 🚀 Features
- Implemented **Multinomial Naive Bayes classifier** with a custom **Count Vectorizer** for token frequency representation.  
- Integrated **entropy-based Active Learning** to iteratively select the most uncertain samples for labeling.  
- Achieved **≈78% accuracy** while using only **27% labeled data** (a **72.9% reduction** in labeling effort).  
- Demonstrated **2× faster retraining** with minimal loss in model performance compared to full supervision.  
- Visualized uncertainty sampling and accuracy trends across iterations.

---

## 🧠 Key Insights
- Active Learning substantially reduces the need for labeled data without degrading accuracy.  
- Entropy-based uncertainty sampling prioritizes informative samples, improving convergence efficiency.  
- Naive Bayes provides strong baseline performance for low-resource text classification tasks.

---

## 🧮 Dataset
- **Source:** Processed subset of the **Sentiment140 dataset**.  
- **Type:** Binary sentiment classification (Positive vs. Negative).  
- **Size:** ~1 million samples (reduced during Active Learning).  

---

## 📈 Results
| Training Setup | Labeled Data Used | Accuracy | Training Time |
|----------------|------------------:|----------:|---------------:|
| Full Supervision | 100% | 78.1% | 1× |
| Active Learning | 27% | 77.9% | **0.5×** |

> Active Learning achieved comparable accuracy with less than one-third of the labeled data.

---

## 🛠️ Requirements
```bash
pip install numpy pandas matplotlib scikit-learn
