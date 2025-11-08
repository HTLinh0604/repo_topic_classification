# 🔬 PROJECT 2B: GitHub Repository Topic Classification Based on Textual Metadata
*(DỰ ÁN 2B: Phân loại Chủ đề Kho lưu trữ GitHub Dựa trên Siêu dữ liệu Văn bản)*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-F08030?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PEFT](https://img.shields.io/badge/PEFT-FFB762?style=for-the-badge&logo=huggingface&logoColor=black)
![NLTK](https://img.shields.io/badge/NLTK-306998?style=for-the-badge&logo=nltk&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GraphQL](https://img.shields.io/badge/GraphQL-E10098?style=for-the-badge&logo=graphql&logoColor=white)

---

## 🎯 Introduction & Goal
*(Giới thiệu & Mục tiêu)*

* **Problem:** The rapid growth of open-source (OSS) repositories on GitHub has generated vast amounts of textual metadata. While GitHub allows users to assign topics, these labels are often **inconsistent, incomplete, and user-generated**.
    *(**Vấn đề:** Sự phát triển nhanh chóng của các kho mã nguồn mở (OSS) trên GitHub đã tạo ra lượng lớn siêu dữ liệu văn bản. Mặc dù GitHub cho phép người dùng gán chủ đề, các nhãn này thường **không nhất quán, không đầy đủ và do người dùng tạo**.)*

* **Goal:** To develop an automated and comprehensive approach to classify repositories into specific topics based on their textual metadata, primarily the content from **`README.md`** files.
    *(**Mục tiêu:** Phát triển một cách tiếp cận tự động và toàn diện để phân loại các kho lưu trữ vào các chủ đề cụ thể dựa trên siêu dữ liệu văn bản, chủ yếu là nội dung từ file **`README.md`**.)*

* **Main Contributions:**
    *(**Đóng góp chính:**)*
    1.  Constructed and publicly described a **large, diverse dataset** of `README.md` files, covering over **fifty different topics**.
        *(Xây dựng và mô tả công khai một **tập dữ liệu lớn, đa dạng** gồm các file `README.md`, bao gồm hơn **năm mươi chủ đề** khác nhau.)*
    2.  Developed a **robust text preprocessing** pipeline tailored for software documentation (e.g., stopword enrichment with domain-specific tokens).
        *(Phát triển quy trình **tiền xử lý văn bản mạnh mẽ** được điều chỉnh cho tài liệu phần mềm (ví dụ: làm giàu stopword bằng các mã thông báo dành riêng cho miền).)*
    3.  Applied and evaluated state-of-the-art Transformer-based models, leveraging **PEFT (LoRA)** to achieve high performance and computational efficiency.
        *(Áp dụng và đánh giá các mô hình dựa trên Transformer hiện đại, tận dụng **PEFT (LoRA)** để đạt được hiệu suất cao và hiệu quả tính toán.)*

---

## 📊 Data & Preprocessing
*(Dữ liệu & Tiền xử lý)*

### A. Data Collection
*(A. Thu thập Dữ liệu)*

* **Source:** Used the **GitHub GraphQL API** for data collection.
    *(**Nguồn:** Sử dụng **GitHub GraphQL API** để thu thập dữ liệu.)*
* **Scale:** The final dataset contains **57,368 `README.md` files**.
    *(**Quy mô:** Tập dữ liệu cuối cùng chứa **57.368 file `README.md`**.)*
* **Topic Scope:** Over 50 distinct topics within the Information Technology (IT) domain.
    *(**Phạm vi Chủ đề:** Hơn 50 chủ đề riêng biệt trong lĩnh vực Công nghệ Thông tin (IT).)*
* **Sampling Strategy (to ensure diversity):** For each topic, up to 2,000 repositories were collected based on 4 complementary criteria (approx. 500 repos each):
    *(**Chiến lược lấy mẫu (để đảm bảo tính đa dạng):** Đối với mỗi chủ đề, thu thập tới 2.000 kho lưu trữ dựa trên 4 tiêu chí bổ sung (mỗi tiêu chí khoảng 500 repo):)*
    1.  **Most Starred**
        *(**Most Starred** (Được gắn sao nhiều nhất).)*
    2.  **Most Forked**
        *(**Most Forked** (Được fork nhiều nhất).)*
    3.  **Recently Updated**
        *(**Recently Updated** (Được cập nhật gần đây).)*
    4.  **Best Match** (Random sampling for diversity)
        *(**Best Match** (Mẫu ngẫu nhiên để tăng tính đa dạng).)*
* **Target Labels:** 50 granular topics were mapped to **10 broader categories** representing major IT domains, serving as the target labels for the multi-class classification task.
    *(**Phân loại Mục tiêu (Target Labels):** 50 chủ đề chi tiết đã được ánh xạ thành **10 danh mục rộng hơn** (categories) đại diện cho các miền chính trong IT, dùng làm nhãn mục tiêu cho nhiệm vụ phân loại đa lớp.)*
* **Data Distribution:** The dataset was split using **stratified sampling** with an **80% Train** and **20% Test** ratio. Total 57,368 samples, with 45,894 training and 11,474 test samples.
    *(**Phân phối Dữ liệu:** Tập dữ liệu được phân chia theo **lấy mẫu phân tầng (stratified sampling)** với tỷ lệ **80% cho tập huấn luyện (Train)** và **20% cho tập kiểm tra (Test)**. Tổng cộng 57.368 mẫu, với 45.894 mẫu huấn luyện và 11.474 mẫu kiểm tra.)*

### B. Preprocessing Pipeline
*(B. Quy trình Tiền xử lý)*

A custom preprocessing pipeline was applied to remove noise typical in software documentation:
*(Một pipeline tiền xử lý tùy chỉnh đã được áp dụng để loại bỏ nhiễu điển hình trong tài liệu phần mềm:)*

1.  **Code and URL Removal:** Stripped Markdown code blocks, inline code snippets, and hyperlinks.
    *(**Loại bỏ Code và URL:** Xóa các khối mã Markdown, đoạn mã nội tuyến và siêu liên kết.)*
2.  **Markdown Syntax Cleaning:** Normalized headers, bold/italic formatting to plain text.
    *(**Làm sạch Cú pháp Markdown:** Chuẩn hóa các dấu tiêu đề, định dạng in đậm/in nghiêng thành văn bản thuần túY.)*
3.  **Tokenization and Lemmatization:** Used `NLTK` to reduce word inflections.
    *(**Tokenization và Lemmatization:** Sử dụng `NLTK` để giảm các dạng biến tố của từ.)*
4.  **Stopword Filtering:** Extended the standard English stopword list with a custom list of programming-related terms (e.g., *install, build, repository, command, example, file*) to reduce uninformative content.
    *(**Lọc Stopword:** Mở rộng danh sách stopword tiếng Anh tiêu chuẩn bằng một danh sách các thuật ngữ liên quan đến lập trình (ví dụ: *install, build, repository, command, example, file*) để giảm thiểu nội dung không mang thông tin.)*
5.  **Lowercasing and Noise Removal:** All text was lowercased and cleaned of punctuation, digits, and special characters.
    *(**Chuyển về chữ thường và loại bỏ nhiễu:** Tất cả văn bản được chuyển về chữ thường và làm sạch khỏi dấu câu, chữ số và ký tự đặc biệt.)*

---

## 🛠️ Methodology & Models
*(Phương pháp & Kiến trúc Mô hình)*

### A. Feature Representation
*(A. Biểu diễn Đặc trưng)*

* **For Classical Models:** Utilized **Sentence-BERT** embeddings (specifically the **all-MiniLM-L6-v2** model) to convert each preprocessed README into a dense **384-dimensional** feature vector.
    *(**Đối với Mô hình Cổ điển:** Sử dụng nhúng câu **Sentence-BERT** (cụ thể là mô hình **all-MiniLM-L6-v2**) để chuyển đổi mỗi file README đã được tiền xử lý thành một vector đặc trưng dày đặc **384 chiều**.)*
* **For Transformer Models:** Used the **AutoTokenizer** corresponding to the base model **Mistral-7B-v0.1**. Each text was encoded into a fixed-length sequence of **512 tokens**.
    *(**Đối với Mô hình Transformer:** Sử dụng **AutoTokenizer** tương ứng với mô hình cơ sở **Mistral-7B-v0.1**. Mỗi văn bản được mã hóa thành một chuỗi ID token có độ dài cố định **512 tokens**.)*

### B. Models Compared
*(B. Mô hình được So sánh)*

1.  **Classical Machine Learning Models:** Trained on MiniLM embeddings:
    *(**Mô hình Học máy Cổ điển:** Huấn luyện trên các nhúng MiniLM:)*
    * Logistic Regression (LR)
    * Random Forest (RF)
    * Support Vector Classifier (SVC) (using a linear kernel)
    * K-Nearest Neighbors (KNN)
2.  **Modern Transformer Model (PEFT):**
    *(**Mô hình Transformer Hiện đại (PEFT):**)*
    * **Mistral-7B** fine-tuned using **Low-Rank Adaptation (LoRA)** and **4-bit Quantization**.
        *(**Mistral-7B** được tinh chỉnh bằng cách sử dụng **Low-Rank Adaptation (LoRA)** và **Lượng tử hóa 4-bit**.)*
    * The PEFT/LoRA technique significantly reduced trainable parameters (using a low rank $r=16$) while maintaining robust model performance.
        *(Kỹ thuật PEFT/LoRA giảm đáng kể các tham số huấn luyện (sử dụng chiều rank thấp $r=16$) trong khi vẫn duy trì hiệu suất mô hình mạnh mẽ.)*

---

## 📈 Experimental Results
*(Kết quả Thực nghiệm)*

Performance was evaluated using Precision (P), Recall (R), F1-score (F1), and Accuracy (A).
*(Hiệu suất được đánh giá bằng Precision (P), Recall (R), F1-score (F1), và Accuracy (A).)*

### A. Classical Model Performance
*(A. Hiệu suất Mô hình Cổ điển)*

The classical machine learning models achieved **moderate results**, indicating a limited ability to capture complex semantic context.
*(Các mô hình học máy cổ điển đạt kết quả **ở mức vừa phải**, cho thấy khả năng hạn chế trong việc nắm bắt bối cảnh ngữ nghĩa phức tạp.)*

| Model | Precision | Recall | **F1-Score** |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.66 | 0.69 | 0.66 |
| Random Forest | 0.58 | 0.62 | **0.56** (Lowest) |
| SVC | 0.66 | 0.69 | 0.67 |
| **KNN** | 0.67 | 0.70 | **0.68** (Highest in classical) |

### B. Transformer Model Performance (Mistral-7B + PEFT/LoRA)
*(B. Hiệu suất Mô hình Transformer (Mistral-7B + PEFT/LoRA))*

The fine-tuned Transformer model achieved **superior performance improvements**.
*(Mô hình Transformer được tinh chỉnh đạt được sự **cải thiện hiệu suất vượt trội**.)*

| Metric Type | Precision | Recall | **F1-score** | **Accuracy** |
| :--- | :--- | :--- | :--- | :--- |
| Per-class range | 0.94–0.97 | 0.92–0.97 | 0.93–0.96 | – |
| **Macro & Weighted Avg** | **0.95** | **0.95** | **0.95** | **0.95** |

* The Mistral-7B model with PEFT/LoRA demonstrated **consistent high performance** across all classes, achieving an **Overall Accuracy of 0.95**.
    *(Mô hình Mistral-7B với PEFT/LoRA thể hiện **hiệu suất cao nhất quán** trên tất cả các lớp, đạt **Độ chính xác tổng thể 0.95**.)*
* Classes such as 8 and 3 achieved the highest Precision and Recall (around 0.97).
    *(Các lớp như 8 và 3 đạt Precision và Recall cao nhất (khoảng 0.97).)*

---

## 🏁 Conclusion
*(Kết luận)*

* **Power of Transformers:** The fine-tuned Mistral-7B model demonstrated **superior representational power** and deeper contextual understanding compared to classical models.
    *(**Sức mạnh của Transformer:** Mô hình Mistral-7B tinh chỉnh đã chứng minh **khả năng biểu diễn vượt trội** và hiểu biết ngữ cảnh sâu sắc hơn so với các mô hình cổ điển.)*
* **Efficacy of PEFT:** The use of **LoRA and 4-bit quantization** confirmed that PEFT provides an effective balance between **high performance and low computational cost** for large-scale repository analysis. This method enables the application of high-capacity Transformers to specialized classification problems without full fine-tuning.
    *(**Hiệu quả của PEFT:** Việc sử dụng **LoRA và lượng tử hóa 4-bit** đã xác nhận rằng PEFT cung cấp sự cân bằng hiệu quả giữa **hiệu suất cao và chi phí tính toán thấp** cho phân tích kho lưu trữ quy mô lớn. Phương pháp này giúp áp dụng các Transformer công suất cao vào các vấn đề phân loại chuyên biệt mà không cần tinh chỉnh toàn bộ.)*

---

### 💡 Analogy
*(Phép Tương tự)*

> You can think of classifying GitHub repository topics like organizing a massive library. Classical models (LR, KNN) are like trying to sort books based only on their length or the frequency of a few keywords (TF-IDF/MiniLM)—you might separate sci-fi from cookbooks, but you'll struggle with complex novels (overlapping topics).
>
> *(Bạn có thể coi việc phân loại chủ đề kho lưu trữ GitHub giống như việc sắp xếp một thư viện khổng lồ. Các mô hình cổ điển (LR, KNN) giống như việc bạn cố gắng sắp xếp sách chỉ dựa trên độ dài hoặc tần suất xuất hiện của một vài từ khóa (TF-IDF/MiniLM)—bạn có thể phân loại được sách khoa học viễn tưởng và sách dạy nấu ăn, nhưng sẽ gặp khó khăn với các tiểu thuyết phức tạp (chủ đề chồng chéo).)*
>
> In contrast, the PEFT-tuned Transformer model is like a **highly trained librarian (Mistral-7B)** who doesn't just read keywords but understands the context, sentence structure, and deep purpose of the book (the README), allowing them to accurately shelve every topic, even the most specialized or hybrid ones, with high speed and efficiency.
>
> *(Ngược lại, mô hình Transformer được tinh chỉnh bằng PEFT giống như một **thủ thư được đào tạo chuyên sâu (Mistral-7B)**, người không chỉ đọc từ khóa mà còn hiểu được bối cảnh, cấu trúc câu và mục đích sâu xa của cuốn sách (tài liệu README), cho phép họ sắp xếp chính xác mọi chủ đề, kể cả những cuốn sách chuyên biệt hoặc lai tạp nhất, với tốc độ và hiệu suất cao.)*
