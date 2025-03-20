# 🎨 Pixel Encryption Tool v2.0

## 🚀 Overview
The **Pixel Encryption Tool** is an exciting way to protect your images with cutting-edge encryption methods. With its interactive interface, you'll enjoy both security and a sleek user experience! 🔐

✨ **Key Highlights:**
- 🔄 **Arnold Cat Map Pixel Scrambling** (Mixes up image pixels like a puzzle)
- 🔒 **AES-256 Encryption** (Ensures high-level security)
- 📂 **Base64 Encoding** (Compact storage format)
- 💻 **Rich Terminal UI** for interactive experience (with fallback for simpler consoles)

---

## 🛠️ Installation
### Prerequisites
✅ Python 3.8+ must be installed on your system.  
✅ Required Libraries:
- `numpy`
- `opencv-python`
- `pycryptodome`
- `rich` (optional for enhanced UI)

### Quick Setup
Run this command to install all dependencies in one go:
```
pip install numpy opencv-python pycryptodome rich
```

### Run the Tool
```
python pixel_encryption.py
```

---

## 📋 How to Use
### 🔹 Main Menu Options
1️⃣ **Encrypt Image** - Scramble and encrypt an image securely.  
2️⃣ **Decrypt Image** - Restore an encrypted image to its original form.  
3️⃣ **View Program Info** - Learn more about the tool.  
4️⃣ **Exit** - Close the program.  

### 🔹 Encryption Process (Step-by-Step)
🖼️ **Input Image Path:** Example: `example.png`  
📂 **Output Image Path:** Example: `encrypted_output.png`  
🔑 **Secure Key:** Enter a memorable key (Minimum 4 characters).  
🔄 **Iterations:** Enter a number (Recommended: 3-10 for strong security).  

### 🔹 Decryption Process
Just follow the same steps but ensure you use the **same key and iterations** as the encryption step. 

> **⚠️ Important:** Incorrect key or iteration count will fail the decryption process.

---

## 🧪 Example Commands
### Encryption Example
```
python pixel_encryption.py
# Select [1] Encrypt Image
# Enter input path: example.png
# Enter output path: encrypted_output.png
# Enter encryption key: mysecurekey
# Enter number of iterations: 5
```

### Decryption Example
```
python pixel_encryption.py
# Select [2] Decrypt Image
# Enter input path: encrypted_output.png
# Enter output path: decrypted_image.png
# Enter encryption key: mysecurekey
# Enter number of iterations: 5
```

---

## 🛑 Troubleshooting
**Common Issues and Solutions:**
❌ **"Missing dependencies detected."**  
✅ Run `pip install -r requirements.txt` to fix it.

❌ **"Encryption error: Invalid image format."**  
✅ Ensure you use supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`

❌ **"Decryption error: Incorrect key or iteration count."**  
✅ Confirm you are using the **same key and iteration count** from the encryption step.

❌ **"Memory Error: Not enough memory."**  
✅ Try resizing the image or closing background apps.

---

## 👨‍💻 Credits
💡 **Developed by:** Ashok (Nickname: NeospectraX)  
🔗 For issues or contributions, visit [GitHub](https://github.com/your-repository)

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to modify and enhance as per your needs.

---

## ⚠️ Disclaimer
This tool is designed for **educational and research purposes only**. Unauthorized or malicious use is strictly prohibited. Always ensure you have permission before performing encryption/decryption on any image or file.

