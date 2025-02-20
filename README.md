# 📱 TensorFlow Lite Application  

🚀 **TensorFlow Lite Application** 是一個基於 TensorFlow Lite（TFLite）開發的應用專案，旨在在移動設備或邊緣設備上運行輕量級機器學習模型。本專案提供完整的 **TFLite 模型部署流程**，並展示如何在 Android 或其他嵌入式設備上進行推理。

---

## 📌 特色
✅ **輕量級**：適用於移動端與低功耗設備，優化推理速度。  
✅ **完整範例**：包含 TensorFlow Lite 模型轉換、部署與應用。  
✅ **跨平台支持**：可在 Android、Raspberry Pi 或其他邊緣設備上運行。  
✅ **開源社群支援**：可自由修改與擴展以適應不同的 AI 應用場景。  

---

## 📂 專案結構
```
Tensorflow_Lite_Application/
│── models/            # 預訓練的 TensorFlow Lite 模型
│── scripts/           # 相關的模型轉換與測試腳本
│── android_app/       # Android 應用程式示例
│── README.md          # 專案說明文件
│── requirements.txt   # 依賴環境安裝需求
```

---

## 🛠️ 環境安裝  

請先確保安裝 **Python 3.7+** 以及 **TensorFlow** 相關套件：

```bash
pip install -r requirements.txt
```

若需在 Android 設備上運行，請確保 **Android Studio 以及 TFLite 依賴項目已安裝**。  

---

## 🚀 使用方法  

### 📌 1. 模型轉換 (TensorFlow → TFLite)
若你有一個訓練好的 TensorFlow 模型，可使用以下腳本進行 TFLite 轉換：
```bash
python scripts/convert_to_tflite.py --model_path saved_model/ --output_path models/model.tflite
```

### 📌 2. 部署到 Android  
1. 將 `models/model.tflite` 放入 `android_app/assets/` 資料夾。  
2. 使用 Android Studio 開啟 `android_app/` 目錄，並編譯執行應用。  
3. 在模擬器或 Android 設備上測試推理結果。

### 📌 3. 在 Raspberry Pi 運行  
1. 安裝 TFLite Runtime：
```bash
pip install tflite-runtime
```
2. 運行測試腳本：
```bash
python scripts/run_tflite.py --model_path models/model.tflite
```

---

