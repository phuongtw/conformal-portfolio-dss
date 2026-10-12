# Các bản thử nghiệm trước

Thư mục này lưu quá trình phát triển, không phải các bước bắt buộc phải chạy trước bản chính.

| Bản | Nội dung |
|---|---|
| `01_data_clean.ipynb` / `.py` | Giá ETF, lợi suất, Ridge và split conformal chuẩn. |
| `02_baseline.ipynb` / `.py` | Thêm minimum variance, robust MV và backtest có phí cho Ridge. |
| `results_baseline/` | Kết quả đã lưu của bản 02, trước khi bổ sung PyTorch và GARCH. |

**Bản chính:** [03_torch_garch_baseline.ipynb](../03_torch_garch_baseline.ipynb). Kết quả hiện tại ở [results_torch_garch/](../results_torch_garch/).

Nếu chạy script cũ, chạy từ thư mục gốc repository để tìm đúng dữ liệu. Bản cũ có thể xuất `results_baseline/` mới ở thư mục gốc; bộ kết quả lịch sử trong thư mục này được giữ để tham khảo.
