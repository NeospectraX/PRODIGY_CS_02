# 🚀 Pixel Encryption Tool

An advanced Python-based tool for securing images through encryption and steganography. This project combines the Arnold Cat Map for pixel scrambling, AES-256 encryption, and Least Significant Bit (LSB) steganography to protect your images. It features a user-friendly CLI with optional rich text formatting.

---

## 🔥 Features
✅ **Image Encryption:**
- Arnold Cat Map for pixel scrambling
- AES-256 encryption with a user-defined key
- Base64 encoding for output

✅ **Steganography:**
- Hides encrypted images inside cover images using LSB technique

✅ **Decryption:**
- Reverses the encryption process with the correct key and iteration count

✅ **Interactive CLI:**
- Neon-colored menu with progress bars (via `rich`, optional)

✅ **Dependency Auto-Install:**
- Automatically checks and installs missing packages

✅ **Error Handling:**
- Robust validation for file paths, keys, and image sizes

---

## 📋 Prerequisites
- Python 3.8 or higher
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`
- Maximum image size: 100MB, dimensions up to 8192x8192 pixels

---

## 💻 Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/NeospectraX/PRODIGY_CS_02.git
cd pixel-encryption-tool
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the tool:**
```bash
python pixel.py
```

---

## 🚀 Usage

1. **Launch the tool:**
```bash
python pixel.py
```

2. **Choose an option from the menu:**

🔹 **Encrypt Image:** Encrypt an image with a key and iteration count  
🔹 **Decrypt Image:** Decrypt an encrypted image  
🔹 **Stego Embed (Hide Image):** Hide an encrypted image inside a cover image  
🔹 **Stego Extract (Reveal):** Extract and decrypt a hidden image  
🔹 **View Program Info:** Display tool information and tips  
🔹 **Exit:** Quit the program  

---

## 📖 Examples

### 🔒 Encrypt an Image
```
Select an option: 1
Enter input image path: input.png
Enter output image path: encrypted.png
Enter encryption key: mysecretkey123
Enter number of iterations (1-100): 5
```

### 🔓 Decrypt an Image
```
Select an option: 2
Enter input image path: encrypted.png
Enter output image path: decrypted.png
Enter encryption key: mysecretkey123
Enter number of iterations (1-100): 5
```

### 🕵️‍♂️ Embed a Secret Image
```
Select an option: 3
Enter input image path: cover.png
Enter output image path: stego.png
Enter encryption key: mysecretkey123
Enter number of iterations (1-100): 5
Enter secret image path: secret.png
```

---

## ⚙️ How It Works

### 🧩 **Encryption Process**
1. Pixels are scrambled using the Arnold Cat Map with a key-derived parameter.  
2. The scrambled image is encrypted with AES-256 in CBC mode.  
3. The result is encoded in Base64 and saved as an image.  

### 🕵️‍♀️ **Steganography Process**
1. The secret image is encrypted first.  
2. The encrypted image is embedded into a cover image using the LSB technique.  

### 🔓 **Decryption Process**
- Reverses the encryption and steganography process using the same key and iteration count.  

---

## ❗ Important Notes

✅ **Key:** Use the same key for encryption and decryption (Minimum length: 4 characters).  
✅ **Iterations:** Must match between encryption and decryption (1-100).  
✅ **Image Size:** For steganography, the cover image should be significantly larger than the secret image.  
✅ **Troubleshooting:** Check the program info (option 5) for tips if decryption fails.  

---

## 📦 Requirements
See `requirements.txt` for the full list of dependencies.

---

## 📝 License
This project is licensed under the **MIT License**.

💬 _Contributions are welcome! Feel free to fork, improve, and submit pull requests._

