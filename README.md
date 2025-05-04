# CSI ESP32 Tool Kit

This project is a set of tools for user authentication based on Channel State Information (CSI) data using an **ESP32** device.

## Description

The project enables the capture, processing, labeling, model training, and authentication of users based on CSI signals. It was developed from the repository [sbrc2024-csi](https://github.com/c2dc/sbrc2024-csi), adapted for use with ESP32 authentication.

---

### 🧠 **How It Works**

The ESP32 collects CSI data while network traffic (e.g., ping) occurs and a user is either present or not in the environment. These data are processed, labeled, and used to train machine learning authentication models.

<p align="center">
  <img src="docs/csi_esp32_diagram.png" width="70%" alt="How CSI works with ESP32">
</p>

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- ESP-IDF environment configured
- CSI firmware running on ESP32 (via ESP-CSI)

---

## File Structure

- `capture_csi.py`: Captures CSI data from ESP32 via UART serial port and saves it as CSV.
- `process_csi.py`: Processes raw CSV, applying normalization and low-pass filters to amplitude and phase signals.
- `label_csi.py`: Labels the captures with user name, environment, and position; also manages the dataset (add/remove/reset).
- `train_csi.py`: Trains machine learning models (Random Forest, SVM, MLP, Logistic Regression) for multiclass, binary (human presence), and specific (user vs empty environment) authentication.
- `auth_csi.py`: Authenticates a new processed capture using all available models and criteria such as centroid distance.

---

## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Capture CSI data

```bash
python capture_csi.py -p /dev/ttyUSB0 -t 60 -o data/data_csi.csv
```

### 3. Process the captured data

```bash
python process_csi.py -i data/data_csi.csv -o data/processed_csi.csv
```

### 4. Add the capture to the labeled dataset

```bash
python label_csi.py add -i data/processed_csi.csv -u giovani -e room -p standing -o dataset/dataset.csv
```

### 5. (Optional) Remove or reset the dataset

```bash
python label_csi.py remove -u giovani -e room -p standing
python label_csi.py reset
```

### 6. Train authentication models

```bash
python train_csi.py
```

### 7. Authenticate a new capture

```bash
python auth_csi.py -i data/processed_csi.csv -u giovani
```

---

## Notes

- Captures labeled as `user=empty` are used as the background environment without human presence.
- The script `auth_csi.py` performs multiple validations and only authenticates if confidence is high enough, considering binary, multiclass, and centroid distance models.

---

## Based on

- [sbrc2024-csi](https://github.com/c2dc/sbrc2024-csi)

---

## License

MIT
