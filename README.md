# NLP - Sentiment Analysis

This project contains Python source code (Jupyter Notebook) to perform text sentiment analysis. It covers the process from data preprocessing and feature extraction to building prediction models using both traditional Machine Learning and Deep Learning (BERT).

## Folder Structure

```
├── SentimentPredict.ipynb        # Main notebook for training and prediction
├── negative_sentiment_data.json  # Negative sample data
├── neutral_sentiment_data.json   # Neutral sample data
├── positive_sentiment_data.json  # Positive sample data
├── README.md                     # Project description file
└── Task 1/                       # Folder containing classification data and scripts to generate data using LLM
    ├── NegativeGenData/
    ├── NeutralGenData/
    └── PositiveGenData/
```

## Notebook Details

### `SentimentPredict.ipynb` (Sentiment Analysis)
This notebook performs the entire process of building a sentiment analysis system.

*   **Goal:** Classify text into 3 sentiment labels: **Positive**, **Negative**, **Neutral**.
*   **Main Steps:**
    *   **Load Data:** Read data from JSON files corresponding to each label.
    *   **Preprocessing:**
        *   Convert text to lowercase.
        *   Remove special characters and extra spaces.
        *   Remove **Stopwords** (but keep important negative words like "not", "never").
    *   **Feature Extraction:**
        *   **Bag of Words (BoW):** Count word frequency.
        *   **TF-IDF:** Evaluate the importance weight of words.
    *   **Train Machine Learning Models:**
        *   Test models: **Naive Bayes**, **Logistic Regression**, **Random Forest**.
        *   Compare effectiveness between BoW and TF-IDF.
    *   **Check Overfitting:**
        *   Compare accuracy on Train and Test sets.
        *   Draw charts to visualize results.
    *   **Train Deep Learning Model (BERT):**
        *   Use pre-trained language model **BERT (bert-base-uncased)**.
        *   Fine-tune the model on the sentiment dataset.
        *   Use Hugging Face `Trainer` API to train and evaluate.
    *   **Prediction:**
        *   Prediction function to check results from the model.

## Installation Requirements

To run this notebook, you need to install the following Python libraries:

```bash
pip install numpy matplotlib scikit-learn nltk transformers torch
```

---

# NLP - Phân loại cảm xúc

Dự án này bao gồm mã nguồn Python (Jupyter Notebook) để thực hiện quy trình phân tích cảm xúc văn bản (Sentiment Analysis), từ tiền xử lý dữ liệu, trích xuất đặc trưng đến xây dựng mô hình dự đoán sử dụng cả Machine Learning truyền thống và Deep Learning (BERT).

## Cấu trúc Thư mục

```
├── SentimentPredict.ipynb        # Notebook chính cho huấn luyện và dự đoán
├── negative_sentiment_data.json  # Dữ liệu mẫu tiêu cực
├── neutral_sentiment_data.json   # Dữ liệu mẫu trung tính
├── positive_sentiment_data.json  # Dữ liệu mẫu tích cực
├── README.md                     # File mô tả dự án
└── Task 1/                       # Thư mục chứa dữ liệu phân loại và script generate data bằng LLM
    ├── NegativeGenData/
    ├── NeutralGenData/
    └── PositiveGenData/
```

## Chi tiết Notebook

### `SentimentPredict.ipynb` (Phân tích Cảm xúc)
Notebook này thực hiện toàn bộ quy trình xây dựng hệ thống phân tích cảm xúc.

*   **Mục tiêu:** Phân loại văn bản thành 3 nhãn cảm xúc: **Positive** (Tích cực), **Negative** (Tiêu cực), **Neutral** (Trung tính).
*   **Các bước chính:**
    *   **Load dữ liệu:** Đọc dữ liệu từ các file JSON tương ứng với từng nhãn.
    *   **Tiền xử lý (Preprocessing):**
        *   Chuyển văn bản về chữ thường.
        *   Loại bỏ ký tự đặc biệt và khoảng trắng thừa.
        *   Loại bỏ **Stopwords** (nhưng giữ lại các từ phủ định quan trọng như "not", "never").
    *   **Trích xuất đặc trưng (Feature Extraction):**
        *   **Bag of Words (BoW):** Đếm tần suất từ.
        *   **TF-IDF:** Đánh giá trọng số quan trọng của từ.
    *   **Huấn luyện mô hình Machine Learning:**
        *   Thử nghiệm các mô hình: **Naive Bayes**, **Logistic Regression**, **Random Forest**.
        *   So sánh hiệu quả giữa BoW và TF-IDF.
    *   **Kiểm tra Overfitting:**
        *   So sánh độ chính xác trên tập Train và Test.
        *   Vẽ biểu đồ trực quan hóa kết quả.
    *   **Huấn luyện mô hình Deep Learning (BERT):**
        *   Sử dụng mô hình ngôn ngữ tiền huấn luyện **BERT (bert-base-uncased)**.
        *   Fine-tune mô hình trên tập dữ liệu cảm xúc.
        *   Sử dụng `Trainer` API của Hugging Face để huấn luyện và đánh giá.
    *   **Dự đoán:**
        *   Hàm dự đoán để kiểm tra kết quả từ mô hình.

## Yêu cầu cài đặt

Để chạy notebook này, cần cài đặt các thư viện Python sau:

```bash
pip install numpy matplotlib scikit-learn nltk transformers torch
```
