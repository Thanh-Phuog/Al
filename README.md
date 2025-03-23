# 🚀 AI Lab với Tersolow & Hugging Face

Đây là bài nộp cuối cùng của dự án AI sử dụng **Tersolow** và **Hugging Face**.

## 🏷 Nhánh trong repo
- **`master`**: Chứa các bài lab  
- **`final`**: Bài kết thúc môn  

---

## 🛠 Cài đặt môi trường

### 1️⃣ Clone repository
```bash
git clone --b final https://github.com/Thanh-Phuog/Al
cd Al
```

### 2️⃣ Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🚀 Chạy ứng dụng

### 🔹 Cách 1: Chạy trực tiếp
```bash
python app.py
```
*Nếu gặp lỗi, hãy sử dụng Cách 2:*

### 🔹 Cách 2: Chạy bằng Flask CLI (khuyến nghị)

#### ✅ Trên Linux/macOS:
```bash
export FLASK_APP=app.py
flask run
```

#### ✅ Trên Windows (cmd):
```bash
set FLASK_APP=app.py
flask run
```

Sau khi chạy, ứng dụng sẽ hoạt động tại:  
🔗 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** 🚀  

---

## 📌 Tính năng AI được tích hợp

### Phân loại hình ảnh sản phẩm (TensorFlow)
- Ứng dụng sử dụng **TensorFlow** để phân loại hình ảnh sản phẩm.
- Dựa trên kết quả phân loại, hệ thống sẽ tự động gán danh mục phù hợp cho sản phẩm.

### Phân tích đánh giá khách hàng (Hugging Face API)
- Sử dụng **Hugging Face API** để phân tích nhận xét của khách hàng.
- Đưa ra đánh giá **tích cực (positive)** hoặc **tiêu cực (negative)** dựa trên nội dung bình luận.

### Hệ thống gợi ý sản phẩm
- Xây dựng hệ thống **gợi ý sản phẩm** dựa trên tương tác của người dùng.
- Sản phẩm được đề xuất theo danh mục của sản phẩm mà khách hàng đã chọn.

---

## 💎 Công nghệ sử dụng
- 🧠 Hugging Face  
- ⚡ Flask (Python)   
- 🤖 TensorFlow  

---

