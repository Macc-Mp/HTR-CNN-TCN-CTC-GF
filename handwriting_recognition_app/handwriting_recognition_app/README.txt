OFFLINE HANDWRITING RECOGNITION APP

FILES
- app.py: Streamlit user interface and CNN-TCN-CTC inference code
- requirements.txt: Python packages needed by the app
- start_phone.py: LAN/QR launcher for opening the app on a phone
- cnn_tcn_gated_fixedlr_v2.pth: your trained checkpoint (copy this file into this folder)

SETUP (Windows PowerShell)
1. Install Python 3.10 or 3.11 (recommended for PyTorch compatibility).
2. Open PowerShell in this folder.
3. Create a virtual environment:
   py -m venv .venv
4. Install dependencies:
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
5. Copy cnn_tcn_gated_fixedlr_v2.pth into this folder.
6. Start the app:
   .\.venv\Scripts\python.exe -m streamlit run app.py
7. Open the local address Streamlit prints (usually http://localhost:8501).

RUN ON YOUR PHONE (Expo Go-style QR code)
1. Keep the PC and phone on the same Wi-Fi network.
2. From the repository root, run:
   .\start_phone.ps1
   (or: .\.venv\Scripts\python.exe .\handwriting_recognition_app\handwriting_recognition_app\start_phone.py)
3. Scan the QR code printed in the terminal with the phone camera.
4. The app opens at http://<PC-IP>:8501 in the phone browser.
- Use a different port with: .\start_phone.ps1 --port 8600
- If the phone cannot connect, allow Python/streamlit through Windows Firewall
  for private networks, and make sure the Wi-Fi network is marked "Private".
- This is a mobile-friendly web app, not a native installable app: the phone
  must stay connected to this PC while it is running.

CHECKPOINT
By default, the app expects cnn_tcn_gated_fixedlr_v2.pth in the same folder as app.py.
If it is stored elsewhere, enter its full path in the app's sidebar.

IMPORTANT NOTES
- The app uses the architecture and 79-character vocabulary from the supplied notebook.
- It uses the notebook's custom-image preprocessing: grayscale, Otsu thresholding,
  resize to height 128 while preserving aspect ratio, and normalization to [-1, 1].
- It loads the checkpoint once. It does not train or fine-tune the model.
- The app is intended for one-line English handwriting input. Long multi-line text is
  not automatically segmented into separate lines.
- Handwriting can be entered on the canvas or uploaded as a PNG, JPEG, BMP, TIFF, or
  WebP image. The upload tab can compare a prediction with an optional expected text.
- Panelist demonstration predictions are not part of the IAM test-set evaluation.
