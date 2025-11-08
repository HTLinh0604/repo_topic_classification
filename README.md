# 🚀 PROJECT 2B: GitHub Repository Topic Classification Based on Textual Metadata  
# Dự án 2B: Phân loại chủ đề kho GitHub dựa trên siêu dữ liệu văn bản  

**🧠 Technologies:** Python · PyTorch · HuggingFace Transformers · PEFT (LoRA) · Scikit-learn · NLTK · GitHub GraphQL API  

---

## 🧩 1. Introduction & Goal  
## Giới thiệu và Mục tiêu Dự án  

**Project Name:** PROJECT 2B: GitHub Repository Topic Classification Based on Textual Metadata.  
**Tên Dự án:** PROJECT 2B: Phân loại chủ đề kho GitHub dựa trên siêu dữ liệu văn bản.  

**Problem:** The rapid growth of open-source repositories on GitHub has produced vast textual metadata. However, user-generated topic tags are often **inconsistent, incomplete, and unreliable**.  
**Vấn đề:** Sự phát triển nhanh của kho mã nguồn mở GitHub tạo ra lượng lớn siêu dữ liệu văn bản, nhưng nhãn chủ đề thường **không nhất quán, không đầy đủ và phụ thuộc người dùng**.  

**Goal:** Develop an **automated and robust classification approach** based on textual metadata, primarily from **`README.md`** files.  
**Mục tiêu:** Phát triển phương pháp **phân loại tự động và mạnh mẽ** dựa trên siêu dữ liệu văn bản, chủ yếu từ file **`README.md`**.  

**Main Contributions:**  
**Đóng góp chính:**  
1️⃣ Construction of a **large, diverse public dataset** with over **50 distinct topics** from README.md files.  
1️⃣ Xây dựng và công khai **tập dữ liệu lớn, đa dạng** gồm hơn **50 chủ đề** khác nhau từ README.md.  
2️⃣ Development of a **domain-specific text preprocessing pipeline**, enriching stopwords with software-related tokens.  
2️⃣ Phát triển **pipeline tiền xử lý đặc thù miền**, mở rộng stopword bằng các từ liên quan đến lập trình.  
3️⃣ Application of **Transformer-based models using PEFT (LoRA)** for high accuracy and computational efficiency.  
3️⃣ Áp dụng **mô hình Transformer với PEFT (LoRA)** nhằm đạt hiệu suất cao và tiết kiệm tài nguyên.  

---

## 💾 2. Data & Preprocessing  
## Dữ liệu và Tiền xử lý  

### A. Data Collection  
### A. Thu thập Dữ liệu  

**Source:** Collected via **GitHub GraphQL API**.  
**Nguồn:** Thu thập qua **GitHub GraphQL API**.  

**Scale:** Contains **57,368 README.md files**.  
**Quy mô:** Gồm **57.368 file README.md**.  

**Topic Coverage:** Over **50 distinct IT-related topics**.  
**Phạm vi Chủ đề:** Hơn **50 chủ đề trong lĩnh vực CNTT**.  

**Sampling Strategy (Diversity Ensured):** For each topic, up to 2,000 repositories were selected based on:  
**Chiến lược lấy mẫu (đảm bảo đa dạng):** Mỗi chủ đề thu thập tối đa 2.000 kho theo các tiêu chí sau:  
1️⃣ Most Starred  
1️⃣ Được gắn sao nhiều nhất  
2️⃣ Most Forked  
2️⃣ Được fork nhiều nhất  
3️⃣ Recently Updated  
3️⃣ Cập nhật gần đây nhất  
4️⃣ Best Match (random sampling)  
4️⃣ Mẫu ngẫu nhiên để tăng tính đa dạng  

**Target Labels:** 50 topics mapped into **10 broader IT categories** for multi-class classification.  
**Phân loại Mục tiêu:** 50 chủ đề ánh xạ thành **10 danh mục IT chính** cho bài toán đa lớp.  

**Data Split:** **80% train / 20% test** using stratified sampling (45,894 train, 11,474 test).  
**Phân phối Dữ liệu:** **80% huấn luyện / 20% kiểm tra** bằng lấy mẫu phân tầng.  

---

### B. Preprocessing Pipeline  
### B. Quy trình Tiền xử lý  

1️⃣ **Remove Code & URLs:** Clean Markdown code blocks, inline code, and hyperlinks.  
1️⃣ **Xóa Code & URL:** Loại bỏ các khối code, mã nội tuyến, liên kết.  

2️⃣ **Normalize Markdown Syntax:** Convert formatting to plain text.  
2️⃣ **Chuẩn hóa cú pháp Markdown:** Chuyển định dạng về văn bản thuần.  

3️⃣ **Tokenization & Lemmatization:** Applied via `NLTK`.  
3️⃣ **Tokenization & Lemmatization:** Thực hiện bằng `NLTK`.  

4️⃣ **Custom Stopwords:** Extend with domain-specific tokens (*install, build, repository, file*...).  
4️⃣ **Stopword mở rộng:** Bổ sung các từ kỹ thuật (*install, build, repository, file*...).  

5️⃣ **Lowercasing & Noise Removal:** Remove digits, punctuation, special symbols.  
5️⃣ **Chuyển chữ thường & loại nhiễu:** Xóa ký tự đặc biệt, số, dấu câu.  

---

## ⚙️ 3. Methodology & Models  
## Phương pháp và Kiến trúc Mô hình  

### A. Feature Representation  
### A. Biểu diễn Đặc trưng  

**Classical Models:** Use **Sentence-BERT embeddings (all-MiniLM-L6-v2)** → 384-dimensional vectors.  
**Mô hình cổ điển:** Dùng **Sentence-BERT (all-MiniLM-L6-v2)** → vector đặc trưng 384 chiều.  

**Transformer Models:** Tokenized using **AutoTokenizer (Mistral-7B-v0.1)** with fixed length **512 tokens**.  
**Mô hình Transformer:** Sử dụng **AutoTokenizer (Mistral-7B-v0.1)** với chuỗi **512 tokens** cố định.  

---

### B. Models Compared  
### B. Mô hình Được So sánh  

**Classical Machine Learning Models:** (trained on MiniLM embeddings)  
**Mô hình Học máy Cổ điển:** (huấn luyện trên nhúng MiniLM)  
- Logistic Regression (LR)  
- Random Forest (RF)  
- Support Vector Classifier (SVC)  
- K-Nearest Neighbors (KNN)  

**Modern Transformer (PEFT):**  
**Mô hình Transformer Hiện đại (PEFT):**  
- **Mistral-7B** fine-tuned with **Low-Rank Adaptation (LoRA)** and **4-bit quantization**.  
- Reduces trainable parameters ($r=16$) while preserving performance.  
- **Mistral-7B** tinh chỉnh với **LoRA** và **lượng tử hóa 4-bit**, giảm tham số mà vẫn giữ hiệu năng.  

---

## 📊 4. Experimental Results  
## Kết quả Thực nghiệm  

**Metrics:** Precision (P), Recall (R), F1-score (F1), Accuracy (A).  
**Chỉ số đánh giá:** Precision (P), Recall (R), F1-score (F1), Accuracy (A).  

### A. Classical Models Performance  
### A. Hiệu suất Mô hình Cổ điển  

| Model | Precision | Recall | **F1-Score** |  
| :--- | :--- | :--- | :--- |  
| Logistic Regression | 0.66 | 0.69 | 0.66 |  
| Random Forest | 0.58 | 0.62 | **0.56** (Lowest) |  
| SVC | 0.66 | 0.69 | 0.67 |  
| **KNN** | 0.67 | 0.70 | **0.68** (Best) |  

**Observation:** Classical models capture basic text features but fail to represent complex semantics.  
**Nhận xét:** Mô hình cổ điển nắm bắt đặc trưng cơ bản nhưng chưa thể hiện được ngữ nghĩa phức tạp.  

---

### B. Transformer Model (Mistral-7B + PEFT/LoRA)  
### B. Mô hình Transformer (Mistral-7B + PEFT/LoRA)  

| Metric Type | Precision | Recall | **F1-score** | **Accuracy** |  
| :--- | :--- | :--- | :--- | :--- |  
| Per-class Range | 0.94–0.97 | 0.92–0.97 | 0.93–0.96 | – |  
| **Macro / Weighted Avg.** | **0.95** | **0.95** | **0.95** | **0.95** |  

**Observation:** Mistral-7B (PEFT/LoRA) achieved **state-of-the-art performance**, with **consistent accuracy of 0.95** across all classes.  
**Nhận xét:** Mistral-7B (PEFT/LoRA) đạt **hiệu suất vượt trội**, với **độ chính xác 0.95** ổn định trên mọi lớp.  

---

## 🧠 5. Conclusion  
## Kết luận  

✅ **Transformer Strength:** Fine-tuned Mistral-7B demonstrates superior contextual understanding compared to classical ML.  
✅ **Sức mạnh của Transformer:** Mistral-7B tinh chỉnh thể hiện khả năng hiểu ngữ cảnh vượt trội so với mô hình cổ điển.  

⚙️ **Efficiency of PEFT:** LoRA and 4-bit quantization enable **high accuracy with minimal computation cost**, suitable for large-scale repository analysis.  
⚙️ **Hiệu quả của PEFT:** LoRA và lượng tử hóa 4-bit mang lại **hiệu suất cao, chi phí thấp**, phù hợp cho bài toán quy mô lớn.  

---

## 📚 Analogy  
## So sánh Minh họa  

Classical models (LR, KNN) resemble sorting books by keywords — effective for simple topics but weak for nuanced meaning.  
Mô hình cổ điển giống việc phân loại sách bằng từ khóa — hiệu quả với chủ đề đơn giản nhưng kém với ngữ cảnh phức tạp.  

Fine-tuned Transformers like **Mistral-7B + PEFT** act as **expert librarians** who truly understand context and semantics, achieving precise categorization even in ambiguous cases.  
Transformer tinh chỉnh như **Mistral-7B + PEFT** giống **thủ thư chuyên nghiệp**, hiểu sâu ngữ nghĩa và phân loại chính xác cả trường hợp mơ hồ.  
