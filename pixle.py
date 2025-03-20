import os
import sys
import subprocess
import time
import numpy as np
import cv2
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import hashlib
import base64
import pickle

# Try to import rich, if not available we'll use standard output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback print function
    class FallbackConsole:
        def print(self, text):
            # Remove rich formatting
            text = text.replace("[bold red]", "").replace("[/bold red]", "")
            text = text.replace("[bold green]", "").replace("[/bold green]", "")
            text = text.replace("[bold yellow]", "").replace("[/bold yellow]", "")
            text = text.replace("[bold magenta]", "").replace("[/bold magenta]", "")
            text = text.replace("[bold cyan]", "").replace("[/bold cyan]", "")
            text = text.replace("[green]", "").replace("[/green]", "")
            text = text.replace("[yellow]", "").replace("[/yellow]", "")
            text = text.replace("[cyan]", "").replace("[/cyan]", "")
            text = text.replace("[red]", "").replace("[/red]", "")
            print(text)
        
        def input(self, prompt):
            # Remove rich formatting
            prompt = prompt.replace("[bold magenta]", "").replace("[/bold magenta]", "")
            prompt = prompt.replace("[bold cyan]", "").replace("[/bold cyan]", "")
            return input(prompt)
    
    console = FallbackConsole()

def install(package):
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}!")
        return False

def check_dependencies():
    required_packages = ["numpy", "Pillow", "opencv-python", "pycryptodome", "rich"]
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "Pillow":
                __import__("PIL")
            elif package == "opencv-python":
                __import__("cv2")
            elif package == "pycryptodome":
                __import__("Crypto")
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing dependencies detected.")
        all_installed = True
        for package in missing_packages:
            if not install(package):
                all_installed = False
        
        if all_installed:
            print("All dependencies installed. Restarting script...")
            # Fix: Use the correct path for restart
            script_path = os.path.abspath(sys.argv[0])
            os.execv(sys.executable, [sys.executable, script_path])
        else:
            print("Failed to install all dependencies. Please install them manually.")
            sys.exit(1)

# Run dependency check
check_dependencies()

def arnold_cat_map(img, iterations, key):
    """Apply Arnold Cat Map for image scrambling"""
    h, w = img.shape[0], img.shape[1]
    
    # Create a copy to avoid modifying the original
    result = np.copy(img)
    
    for _ in range(iterations):
        temp = np.copy(result)
        for y in range(h):
            for x in range(w):
                # Modified Arnold Cat Map with key parameter
                new_x = (x + key * y) % w
                new_y = (x + (key + 1) * y) % h
                result[new_y, new_x] = temp[y, x]
    
    return result

def inverse_arnold_cat_map(img, iterations, key):
    """Apply inverse Arnold Cat Map to unscramble the image"""
    h, w = img.shape[0], img.shape[1]
    
    # Create a copy to avoid modifying the original
    result = np.copy(img)
    
    for _ in range(iterations):
        temp = np.copy(result)
        for y in range(h):
            for x in range(w):
                # Inverse transformation of Arnold Cat Map with key
                det = (key + 1) - key * 1
                inv_x = ((key + 1) * x - key * y) % w
                inv_y = (-1 * x + y) % h
                result[inv_y, inv_x] = temp[y, x]
    
    return result

def derive_key(key_str, size=32):
    """Derive a consistent key of specified size from a string"""
    return hashlib.sha256(key_str.encode()).digest()[:size]

def encrypt_image(img, key_str, iterations):
    """Main encryption function"""
    try:
        # Ensure image is the right shape and type
        if len(img.shape) == 3 and img.shape[2] == 4:  # Has alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        # Derive a cryptographic key from the string
        key = derive_key(key_str)
        key_int = int.from_bytes(key[:4], 'big') % 8 + 1  # Key for Arnold map (1-8)
        
        # Step 1: Apply Arnold Cat Map scrambling
        scrambled = arnold_cat_map(img, iterations, key_int)
        
        # Step 2: Convert image to bytes and encrypt with AES
        img_bytes = pickle.dumps((scrambled.shape, scrambled.dtype, scrambled.tobytes()))
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted_bytes = iv + cipher.encrypt(pad(img_bytes, AES.block_size))
        
        # Step 3: Store encryption metadata
        metadata = {
            'shape': img.shape,
            'encrypted': True,
            'key_hash': hashlib.sha256(key_str.encode()).hexdigest()[:8],
            'iterations': iterations
        }
        
        # Step 4: Create output array
        meta_bytes = pickle.dumps(metadata)
        
        # Step 5: Combine metadata and encrypted image
        output = base64.b64encode(len(meta_bytes).to_bytes(4, 'big') + meta_bytes + encrypted_bytes)
        output_array = np.frombuffer(output, dtype=np.uint8)
        
        # Reshape to something that can be saved as an image
        sqrt_size = int(np.ceil(np.sqrt(len(output_array))))
        padding = sqrt_size * sqrt_size - len(output_array)
        padded_array = np.pad(output_array, (0, padding), 'constant')
        output_img = padded_array.reshape(sqrt_size, sqrt_size)
        
        # Make it 3-channel for saving as color image
        if len(output_img.shape) == 2:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
            
        return output_img, metadata
        
    except Exception as e:
        console.print(f"Encryption error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def decrypt_image(img, key_str, iterations):
    """Main decryption function"""
    try:
        # Extract the bytes from the image
        img_flat = img.flatten()
        
        # Handle grayscale images
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert to grayscale bytes (take only one channel)
            img_flat = img[:,:,0].flatten()
        
        # Remove padding zeros
        nonzero_indices = np.nonzero(img_flat)[0]
        if len(nonzero_indices) == 0:
            raise ValueError("Invalid encrypted image - no data found")
        
        last_nonzero = nonzero_indices[-1]
        img_flat = img_flat[:last_nonzero+1]
        
        # Base64 decode the data
        try:
            decoded_data = base64.b64decode(img_flat)
        except Exception:
            raise ValueError("Not a valid encrypted image")
        
        # Extract metadata length and metadata
        meta_len = int.from_bytes(decoded_data[:4], 'big')
        metadata = pickle.loads(decoded_data[4:4+meta_len])
        
        # Verify this is an encrypted image
        if not metadata.get('encrypted', False):
            raise ValueError("Not an encrypted image")
        
        # Check if the key matches
        key = derive_key(key_str)
        if metadata.get('key_hash') != hashlib.sha256(key_str.encode()).hexdigest()[:8]:
            console.print("Warning: Key hash doesn't match. Decryption may fail.")
        
        # Extract encrypted image data
        encrypted_bytes = decoded_data[4+meta_len:]
        
        # Decrypt the image bytes
        iv = encrypted_bytes[:16]
        ct = encrypted_bytes[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_bytes = unpad(cipher.decrypt(ct), AES.block_size)
        
        # Reconstruct the image
        shape, dtype, img_bytes = pickle.loads(decrypted_bytes)
        scrambled = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
        
        # Derive the same key for Arnold map
        key_int = int.from_bytes(key[:4], 'big') % 8 + 1
        
        # Apply inverse Arnold Cat Map to unscramble
        unscrambled = inverse_arnold_cat_map(scrambled, iterations, key_int)
        
        return unscrambled
        
    except Exception as e:
        console.print(f"Decryption error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def stego_embed(cover_img, secret_img, key_str, iterations):
    """Hide one image within another using LSB steganography"""
    try:
        # Encrypt the secret image first
        encrypted_secret, _ = encrypt_image(secret_img, key_str, iterations)
        if encrypted_secret is None:
            raise ValueError("Failed to encrypt secret image")
        
        # Resize encrypted data to fit within cover if needed
        if encrypted_secret.size > cover_img.size * 0.9:
            raise ValueError("Secret image too large for this cover image")
            
        # Flatten both images
        cover_flat = cover_img.flatten()
        secret_flat = encrypted_secret.flatten()
        
        # Prepare bit planes
        secret_bits = np.unpackbits(secret_flat)
        
        # Create a copy of cover image
        stego_img = cover_img.copy()
        stego_flat = stego_img.flatten()
        
        # Embed secret size at the beginning
        secret_size = len(secret_bits)
        size_bits = np.unpackbits(np.array([secret_size], dtype=np.uint32).view(np.uint8))
        
        # Embed size bits
        for i in range(len(size_bits)):
            if i < len(stego_flat):
                stego_flat[i] = (stego_flat[i] & 0xFE) | size_bits[i]
        
        # Embed secret bits
        start_idx = len(size_bits)
        for i in range(len(secret_bits)):
            if start_idx + i < len(stego_flat):
                stego_flat[start_idx + i] = (stego_flat[start_idx + i] & 0xFE) | secret_bits[i]
                
        return stego_flat.reshape(cover_img.shape)
        
    except Exception as e:
        console.print(f"Steganography embedding error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def stego_extract(stego_img, key_str, iterations):
    """Extract and decrypt a hidden image"""
    try:
        # Flatten the stego image
        stego_flat = stego_img.flatten()
        
        # Extract the size bits first
        size_bits = np.bitwise_and(stego_flat[:32], 1)
        secret_size = np.packbits(size_bits).view(np.uint32)[0]
        
        if secret_size > len(stego_flat) * 8:
            raise ValueError("Invalid secret size detected")
            
        # Extract secret bits
        secret_bits = np.bitwise_and(stego_flat[32:32+secret_size], 1)
        
        # Convert bits back to bytes
        secret_bytes = np.packbits(secret_bits)
        
        # Reshape into an image
        sqrt_size = int(np.ceil(np.sqrt(len(secret_bytes))))
        padding = sqrt_size * sqrt_size - len(secret_bytes)
        
        if padding > 0:
            secret_bytes = np.pad(secret_bytes, (0, padding), 'constant')
            
        secret_img = secret_bytes.reshape(sqrt_size, sqrt_size)
        
        # If it's a single channel, convert to 3-channel
        if len(secret_img.shape) == 2:
            secret_img = cv2.cvtColor(secret_img, cv2.COLOR_GRAY2BGR)
            
        # Decrypt the extracted image
        decrypted = decrypt_image(secret_img, key_str, iterations)
        
        return decrypted
        
    except Exception as e:
        console.print(f"Steganography extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_image(input_path, output_path, key, iterations, mode, secret_path=None):
    """Process images based on mode selection"""
    try:
        # Validate file extension
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        if not any(input_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported file format. Please use: {', '.join(valid_extensions)}")

        # Validate input path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' does not exist")
            
        # Validate file size
        file_size = os.path.getsize(input_path) / (1024 * 1024)  # Size in MB
        if file_size > 100:  # 100MB limit
            raise ValueError(f"File too large ({file_size:.1f}MB). Maximum size is 100MB")
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except PermissionError:
                raise PermissionError(f"Unable to create output directory: {output_dir}. Check permissions.")
        
        # Read the input image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image '{input_path}'. File may be corrupted or in an unsupported format.")
        
        # Validate image dimensions
        max_dimension = 8192  # Maximum allowed dimension
        if img.shape[0] > max_dimension or img.shape[1] > max_dimension:
            raise ValueError(f"Image dimensions too large. Maximum allowed dimension is {max_dimension}px")

        # Process based on selected mode
        if mode == "encrypt":
            if RICH_AVAILABLE:
                for step in track(range(3), description="Encrypting..."):
                    if step == 0:
                        time.sleep(0.1)  # Simulate processing
                    elif step == 1:
                        encrypted_img, metadata = encrypt_image(img, key, iterations)
                        if encrypted_img is None:
                            return False
                    else:
                        cv2.imwrite(output_path, encrypted_img)
                console.print(f"[bold green]Image encrypted with key hash: {metadata['key_hash']}[/bold green]")
            else:
                print("Encrypting...")
                encrypted_img, metadata = encrypt_image(img, key, iterations)
                if encrypted_img is None:
                    return False
                cv2.imwrite(output_path, encrypted_img)
                print(f"Image encrypted with key hash: {metadata['key_hash']}")
            
        elif mode == "decrypt":
            if RICH_AVAILABLE:
                for step in track(range(3), description="Decrypting..."):
                    if step == 0:
                        time.sleep(0.1)  # Simulate processing
                    elif step == 1:
                        decrypted_img = decrypt_image(img, key, iterations)
                        if decrypted_img is None:
                            return False
                    else:
                        cv2.imwrite(output_path, decrypted_img)
            else:
                print("Decrypting...")
                decrypted_img = decrypt_image(img, key, iterations)
                if decrypted_img is None:
                    return False
                cv2.imwrite(output_path, decrypted_img)
            
        elif mode == "stego_embed" and secret_path:
            if not os.path.exists(secret_path):
                raise FileNotFoundError(f"Secret file '{secret_path}' does not exist")
                
            secret_img = cv2.imread(secret_path, cv2.IMREAD_COLOR)
            if secret_img is None:
                raise ValueError(f"Secret image '{secret_path}' not found or invalid format.")
                
            if RICH_AVAILABLE:
                for step in track(range(3), description="Embedding secret..."):
                    if step == 0:
                        time.sleep(0.1)  # Simulate processing
                    elif step == 1:
                        stego_img = stego_embed(img, secret_img, key, iterations)
                        if stego_img is None:
                            return False
                    else:
                        cv2.imwrite(output_path, stego_img)
            else:
                print("Embedding secret...")
                stego_img = stego_embed(img, secret_img, key, iterations)
                if stego_img is None:
                    return False
                cv2.imwrite(output_path, stego_img)
            
        elif mode == "stego_extract":
            if RICH_AVAILABLE:
                for step in track(range(3), description="Extracting secret..."):
                    if step == 0:
                        time.sleep(0.1)  # Simulate processing
                    elif step == 1:
                        extracted_img = stego_extract(img, key, iterations)
                        if extracted_img is None:
                            return False
                    else:
                        cv2.imwrite(output_path, extracted_img)
            else:
                print("Extracting secret...")
                extracted_img = stego_extract(img, key, iterations)
                if extracted_img is None:
                    return False
                cv2.imwrite(output_path, extracted_img)
        
        return True
        
    except FileNotFoundError as e:
        console.print(f"[bold red]File Error: {str(e)}[/bold red]")
        console.print("[yellow]Please check if the file exists and the path is correct.[/yellow]")
    except ValueError as e:
        console.print(f"[bold red]Value Error: {str(e)}[/bold red]")
        if "dimensions" in str(e):
            console.print("[yellow]Try resizing your image to smaller dimensions.[/yellow]")
    except PermissionError as e:
        console.print(f"[bold red]Permission Error: {str(e)}[/bold red]")
        console.print("[yellow]Try running the program with appropriate permissions.[/yellow]")
    except cv2.error as e:
        console.print(f"[bold red]OpenCV Error: {str(e)}[/bold red]")
        console.print("[yellow]The image might be corrupted or in an unsupported format.[/yellow]")
    except MemoryError:
        console.print("[bold red]Memory Error: Not enough memory to process this image.[/bold red]")
        console.print("[yellow]Try using a smaller image or closing other applications.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unexpected Error: {str(e)}[/bold red]")
        if RICH_AVAILABLE:
            console.print("[yellow]Detailed error information:[/yellow]")
            import traceback
            console.print(traceback.format_exc())
        console.print("[yellow]Please report this error to the developer.[/yellow]")
    return False

# ANSI color codes for terminal output
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_PURPLE = "\033[95m"  # Neon purple
COLOR_PINK = "\033[91m"    # Neon pink
COLOR_BRIGHT_GREEN = "\033[92m"  # Neon green
COLOR_BRIGHT_YELLOW = "\033[93m"  # Neon yellow
COLOR_BRIGHT_BLUE = "\033[94m"   # Neon blue
COLOR_BRIGHT_CYAN = "\033[96m"   # Neon cyan

# Enhanced ASCII Banner with Animation
def display_banner():
    neon_colors = [COLOR_PURPLE, COLOR_PINK, COLOR_BRIGHT_GREEN, COLOR_BRIGHT_YELLOW, COLOR_BRIGHT_BLUE, COLOR_BRIGHT_CYAN]
    banner = f"""
{neon_colors[0]}    ██████╗ ██╗██╗  ██╗███████╗██╗      {COLOR_RESET}
{neon_colors[1]}    ██╔══██╗██║╚██╗██╔╝██╔════╝██║      {COLOR_RESET}
{neon_colors[2]}    ██████╔╝██║ ╚███╔╝ █████╗  ██║      {COLOR_RESET}
{neon_colors[3]}    ██╔═══╝ ██║ ██╔██╗ ██╔══╝  ██║      {COLOR_RESET}
{neon_colors[4]}    ██║     ██║██╔╝ ██╗███████╗███████╗ {COLOR_RESET}
{neon_colors[5]}    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ {COLOR_RESET}
{COLOR_YELLOW}       Advanced Image Encryption Tool v2.0{COLOR_RESET}
    """
    # Simple animation effect
    for i in range(len(banner)):
        sys.stdout.write(banner[i])
        sys.stdout.flush()
        time.sleep(0.005)  # Faster typing effect
    print()
    # Display developer credit
    print(f"{COLOR_CYAN}{'─' * 50}{COLOR_RESET}")
    print(f"{COLOR_BRIGHT_YELLOW}      Developed by Ashok (Nickname: NeospectraX){COLOR_RESET}")
    print(f"{COLOR_CYAN}{'─' * 50}{COLOR_RESET}\n")

def draw_menu():
    if RICH_AVAILABLE:
        menu_box = [
            "[bold cyan]╔════════════════════════════════════════════╗[/bold cyan]",
            "[bold cyan]║ [bold yellow]Pixel Encryption Menu                      [/bold yellow]║[/bold cyan]",
            "[bold cyan]╠════════════════════════════════════════════╣[/bold cyan]",
            "[bold cyan]║ [bold green][1] Encrypt Image                          [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold green][2] Decrypt Image                          [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold green][3] Stego Embed (Hide Image)               [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold green][4] Stego Extract (Reveal)                 [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold green][5] View Program Info                      [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold red][6] Exit                                   [/bold red]║[/bold cyan]",
            "[bold cyan]╚════════════════════════════════════════════╝[/bold cyan]"
        ]
        for line in menu_box:
            console.print(line)
    else:
        print("╔════════════════════════════════════════════╗")
        print("║ Pixel Encryption Menu                      ║")
        print("╠════════════════════════════════════════════╣")
        print("║ [1] Encrypt Image                          ║")
        print("║ [2] Decrypt Image                          ║")
        print("║ [3] Stego Embed (Hide Image)               ║")
        print("║ [4] Stego Extract (Reveal)                 ║")
        print("║ [5] View Program Info                      ║")
        print("║ [6] Exit                                   ║")
        print("╚════════════════════════════════════════════╝")

def display_info():
    info_text = """
About Pixel Encryption Tool

This tool provides several ways to secure your images:

1. Encryption: Scrambles and encrypts images using:
   - Arnold Cat Map (pixel scrambling)
   - AES-256 encryption
   - Base64 encoding

2. Steganography: Hides encrypted images inside other images
   - Uses LSB (Least Significant Bit) technique
   - Secret image is encrypted before hiding

Important Tips:
- Always remember your encryption key!
- When decrypting, use the same key and iterations used for encryption
- Larger images may take longer to process
- For steganography, cover image should be larger than secret image

Troubleshooting:
- If decryption fails, double-check your key and iteration count
- Ensure you're using the correct encrypted file
- Some image formats may cause issues - PNG is recommended
    """
    if RICH_AVAILABLE:
        console.print(Panel(info_text, title="Program Information"))
    else:
        print("\n" + "="*50)
        print("               PROGRAM INFORMATION")
        print("="*50)
        print(info_text)
        print("="*50)

def main():
    # Clear terminal and display banner
    os.system('cls' if os.name == 'nt' else 'clear')
    display_banner()
    
    while True:
        draw_menu()
        choice = console.input("Select an option (1-6): ")
        
        if choice == "6":
            console.print("Exiting... Stay secure!")
            break
            
        if choice == "5":
            display_info()
            console.input("Press Enter to continue...")
            continue
        
        if choice in ["1", "2", "3", "4"]:
            print("="*40)
            
            # Input validation for image paths
            while True:
                input_path = console.input("Enter input image path: ")
                if not input_path:
                    console.print("Image path cannot be empty.")
                    continue
                if not os.path.exists(input_path):
                    console.print(f"File not found: {input_path}")
                    continue
                break
                
            while True:
                output_path = console.input("Enter output image path: ")
                if not output_path:
                    console.print("Output path cannot be empty.")
                    continue
                break
                
            # Key validation
            while True:
                key = console.input("Enter encryption key: ")
                if not key or len(key) < 4:
                    console.print("Key must be at least 4 characters.")
                    continue
                break
                
            # Iterations validation
            while True:
                iterations = console.input("Enter number of iterations (1-100): ")
                try:
                    iterations = int(iterations)
                    if not 1 <= iterations <= 100:
                        console.print("Iterations must be between 1 and 100")
                        continue
                    break
                except ValueError:
                    console.print("Please enter a valid number")
            
            secret_path = None
            if choice == "3":
                while True:
                    secret_path = console.input("Enter secret image path: ")
                    if not os.path.exists(secret_path):
                        console.print(f"File not found: {secret_path}")
                        continue
                    break

            modes = {"1": "encrypt", "2": "decrypt", "3": "stego_embed", "4": "stego_extract"}
            console.print("Processing... Please wait...")
            
            success = process_image(input_path, output_path, key, iterations, modes[choice], secret_path)
            
            if success:
                console.print(f"Operation completed! Saved to {output_path}")
            else:
                console.print(f"Operation failed. Please check the inputs and try again.")
                
            print("="*40)
            console.input("Press Enter to continue...")
        else:
            console.print("Invalid option. Try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()