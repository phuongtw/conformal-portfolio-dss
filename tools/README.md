# Công cụ hỗ trợ

Chạy từ thư mục gốc repository:

```bash
python tools/verify_torch_garch.py
```

Kiểm tra thời điểm dự báo GARCH, đổi đơn vị, không dùng dữ liệu tương lai, ràng buộc tỷ trọng và công thức phí.

Hai builder dùng khi thay đổi nguồn code:

```bash
python tools/build_torch_garch_notebook.py
python tools/build_baseline_notebook.py
```

Builder tạo lại notebook và script, đồng thời **xóa output đã lưu**. Không cần chạy builder để đọc hay chạy notebook chính hiện có.
