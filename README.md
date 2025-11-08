# 🚀 PROJECT 2B: GitHub Repository Topic Classification Based on Textual Metadata  
*(Dự án 2B: Phân loại chủ đề kho GitHub dựa trên siêu dữ liệu văn bản)*  

**🧠 Technologies:** Python · PyTorch · HuggingFace Transformers · PEFT (LoRA) · Scikit-learn · NLTK · GitHub GraphQL API  

---

## 🧩 1. Introduction & Goal  
*(Giới thiệu và Mục tiêu Dự án)*  

**Project Name:** PROJECT 2B: GitHub Repository Topic Classification Based on Textual Metadata.  
*(Tên Dự án: Phân loại chủ đề kho GitHub dựa trên siêu dữ liệu văn bản.)*  

**Problem:** The rapid growth of open-source repositories on GitHub has generated vast textual metadata, but user-created topic tags are often **inconsistent, incomplete, and unreliable**.  
*(Vấn đề: Sự phát triển nhanh của kho mã nguồn mở GitHub tạo ra lượng lớn siêu dữ liệu văn bản, nhưng nhãn chủ đề thường **không nhất quán, không đầy đủ và phụ thuộc người dùng**.)*  

**Goal:** Develop an **automated and robust classification approach** based on textual metadata, primarily extracted from **`README.md`** files.  
*(Mục tiêu: Phát triển phương pháp **phân loại tự động và mạnh mẽ** dựa trên siêu dữ liệu văn bản, chủ yếu từ file **`README.md`**.)*  

**Main Contributions:**  
*(Đóng góp chính:)*  
1️⃣ Build a **large, diverse public dataset** with over **50 unique topics** from README.md files.  
*(Xây dựng và công khai **tập dữ liệu lớn, đa dạng** gồm hơn **50 chủ đề** khác nhau.)*  
2️⃣ Develop a **domain-specific text preprocessing pipeline**, enriching stopwords with software-related tokens.  
*(Phát triển **pipeline tiền xử lý đặc thù miền**, mở rộng stopword bằng các từ liên quan đến lập trình.)*  
3️⃣ Apply **Transformer-based models using PEFT (LoRA)** for high accuracy and efficiency.  
*(Áp dụng **mô hình Transformer với PEFT (LoRA)** nhằm đạt hiệu suất cao và tiết kiệm tài nguyên.)*  

---

## 📚 2. Data & Preprocessing  
*(Dữ liệu và Tiền xử lý)*  

### A. Data Collection  
*(Thu thập Dữ liệu)*  

**Source:** Collected via **GitHub GraphQL API**.  
*(Nguồn: Thu thập qua GitHub GraphQL API.)*  

**Scale:** Contains **57,368 README.md files**.  
*(Quy mô: Gồm 57.368 file README.md.)*  

**Topic Coverage:** Over **50 distinct IT-related topics**.  
*(Phạm vi: Hơn 50 chủ đề thuộc lĩnh vực CNTT.)*  

**Sampling Strategy:** For each topic, up to 2,000 repositories were selected based on:  
*(Chiến lược lấy mẫu: Mỗi chủ đề thu thập tối đa 2.000 kho dựa trên các tiêu chí:)*  
1️⃣ Most Starred  
*(Được gắn sao nhiều nhất)*  
2️⃣ Most Forked  
*(Được fork nhiều nhất)*  
3️⃣ Recently Updated  
*(Cập nhật gần đây nhất)*  
4️⃣ Best Match (random sampling)  
*(Mẫu ngẫu nhiên để tăng tính đa dạng)*  

**Target Labels:** 50 topics mapped into **10 broader IT categories** for multi-class classification.  
*(Phân loại Mục tiêu: 50 chủ đề ánh xạ thành 10 danh mục chính trong IT để phục vụ bài toán phân loại đa lớp.)*  

**Data Split:** **80% training / 20% testing**, stratified by topic (45,894 train, 11,474 test).  
*(Phân phối Dữ liệu: 80% huấn luyện, 20% kiểm tra, chia theo phương pháp phân tầng.)*  

---

### B. Preprocessing Pipeline  
*(Quy trình Tiền xử lý)*  

1️⃣ **Remove Code & URLs:** Clean Markdown code blocks, inline code, and hyperlinks.  
*(Xóa các khối code, mã nội tuyến và liên kết.)*  

2️⃣ **Normalize Markdown Syntax:** Convert formatting to plain text.  
*(Chuẩn hóa cú pháp Markdown về dạng văn bản thuần.)*  

3️⃣ **Tokenization & Lemmatization:** Performed using `NLTK`.  
*(Thực hiện tách và chuẩn hóa từ bằng NLTK.)*  

4️⃣ **Custom Stopwords:** Extend with programming-related terms (*install, build, repository, file*...).  
*(Mở rộng danh sách stopword bằng các thuật ngữ lập trình.)*  

5️⃣ **Lowercasing & Noise Removal:** Remove digits, punctuation, and special symbols.  
*(Chuyển toàn bộ về chữ thường và loại bỏ ký tự nhiễu, số, dấu câu.)*  

---

## ⚙️ 3. Methodology & Models  
*(Phương pháp và Kiến trúc Mô hình)*  

### A. Feature Representation  
*(Biểu diễn Đặc trưng)*  

**Classical Models:** Use **Sentence-BERT embeddings (all-MiniLM-L6-v2)** to convert text into 384-dimensional dense vectors.  
*(Mô hình cổ điển: Sử dụng Sentence-BERT (all-MiniLM-L6-v2) để biểu diễn văn bản thành vector 384 chiều.)*  

**Transformer Models:** Use **AutoTokenizer (Mistral-7B-v0.1)** with a fixed input length of **512 tokens**.  
*(Mô hình Transformer: Sử dụng AutoTokenizer (Mistral-7B-v0.1) với độ dài đầu vào cố định 512 tokens.)*  

---

### B. Models Compared  
*(Các Mô hình So sánh)*  

**Classical Machine Learning Models:** (trained on MiniLM embeddings)  
*(Mô hình Học máy Cổ điển: huấn luyện trên nhúng MiniLM)*  
- Logistic Regression (LR)  
- Random Forest (RF)  
- Support Vector Classifier (SVC)  
- K-Nearest Neighbors (KNN)  

**Modern Transformer (PEFT):**  
*(Mô hình Transformer Hiện đại với PEFT)*  
- **Mistral-7B** fine-tuned with **Low-Rank Adaptation (LoRA)** and **4-bit quantization**.  
*(Mistral-7B được tinh chỉnh bằng LoRA và lượng tử hóa 4-bit.)*  
- Reduces trainable parameters ($r=16$) while maintaining performance.  
*(Giảm tham số huấn luyện đáng kể nhưng vẫn đảm bảo hiệu suất cao.)*  

---

## 📊 4. Experimental Results  
*(Kết quả Thực nghiệm)*  

**Metrics:** Precision (P), Recall (R), F1-score (F1), and Accuracy (A).  
*(Chỉ số đánh giá: Precision, Recall, F1-score, Accuracy.)*  

### A. Classical Models Performance  
*(Hiệu suất Mô hình Cổ điển)*  

| Model | Precision | Recall | **F1-Score** |  
| :--- | :--- | :--- | :--- |  
| Logistic Regression | 0.66 | 0.69 | 0.66 |  
| Random Forest | 0.58 | 0.62 | **0.56** (Lowest) |  
| SVC | 0.66 | 0.69 | 0.67 |  
| **KNN** | 0.67 | 0.70 | **0.68** (Best) |  

*(Nhận xét: Các mô hình học máy cổ điển đạt kết quả trung bình, hạn chế trong việc nắm bắt ngữ nghĩa phức tạp.)*  

---

### B. Transformer Model (Mistral-7B + PEFT/LoRA)  
*(Mô hình Transformer: Mistral-7B + PEFT/LoRA)*  

| Metric Type | Precision | Recall | **F1-score** | **Accuracy** |  
| :--- | :--- | :--- | :--- | :--- |  
| Per-class Range | 0.94–0.97 | 0.92–0.97 | 0.93–0.96 | – |  
| **Macro / Weighted Avg.** | **0.95** | **0.95** | **0.95** | **0.95** |  

*(Nhận xét: Mistral-7B (PEFT/LoRA) đạt hiệu suất cao nhất với độ chính xác 0.95 ổn định trên mọi lớp.)*  

---

## 🧠 5. Conclusion  
*(Kết luận)*  

✅ **Transformer Strength:** Fine-tuned Mistral-7B exhibits superior contextual understanding compared to traditional ML models.  
*(Mistral-7B tinh chỉnh thể hiện khả năng hiểu ngữ cảnh vượt trội so với mô hình cổ điển.)*  

⚙️ **Efficiency of PEFT:** LoRA and 4-bit quantization balance **high performance with low computational cost**, making it suitable for large-scale GitHub repository analysis.  
*(LoRA và lượng tử hóa 4-bit mang lại hiệu suất cao với chi phí tính toán thấp, phù hợp cho phân tích quy mô lớn.)*  

---

## 📚 Analogy  
*(So sánh Minh họa)*  

Classical models (LR, KNN) resemble sorting books by keywords — they work for simple distinctions but fail for complex semantics.  
*(Mô hình cổ điển giống việc phân loại sách bằng từ khóa — hiệu quả với chủ đề đơn giản nhưng kém với ngữ nghĩa phức tạp.)*  

Fine-tuned Transformers like **Mistral-7B + PEFT** act as **expert librarians**, understanding context and semantics deeply to classify even nuanced topics accurately.  
*(Transformer tinh chỉnh như Mistral-7B + PEFT giống như thủ thư chuyên nghiệp, hiểu sâu ngữ nghĩa và phân loại chính xác cả các chủ đề phức tạp.)*  
