# AI Cancer Shield

A Streamlit platform for cancer awareness and early detection. After signing in, users can upload medical scans for an AI prediction or ask a cancer-information chatbot questions.

## Features

- **Brain cancer prediction:** classifies uploaded MRI images with a fine-tuned EfficientNet-B0 model (PyTorch).
- **Breast cancer prediction:** classifies uploaded breast-scan images with a second EfficientNet-B0 model.
- **Cancer Q&A chatbot:** matches a question to the closest entry in `finaldata.csv` (TF-IDF and cosine similarity) and answers with a Hugging Face question-answering pipeline.
- **User accounts:** simple sign-up and login, stored in a local SQLite database (`data.db`) with SHA-256 password hashes.

## Tech stack

Python, Streamlit, PyTorch / torchvision, Hugging Face Transformers, scikit-learn, pandas, SQLite

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Entry point: login/sign-up and page navigation |
| `predict_cancer.py` | Brain cancer image prediction |
| `breast_cancer.py` | Breast cancer image prediction |
| `chatbot.py` | Cancer information chatbot |
| `Brain Cancer_efficientnet_best.pth`, `Breast Cancer_efficientnet_best.pth` | Trained model weights |
| `finaldata.csv` | Question/answer data for the chatbot |

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

> This project is for education and research. It is not a medical diagnostic tool.
