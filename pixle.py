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
    class FallbackConsole:
        def print(self, text):
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
            script_path = os.path.abspath(sys.argv[0])
            os.execv(sys.executable, [sys.executable, script_path])
        else:
            print("Failed to install all dependencies. Please install them manually.")
            sys.exit(1)

check_dependencies()

def arnold_cat_map(img, iterations, key):
    """Apply Arnold Cat Map for image scrambling"""
    h, w = img.shape[0], img.shape[1]
    result = np.copy(img)
    
    for _ in range(iterations):
        temp = np.copy(result)
        for y in range(h):
            for x in range(w):
                new_x = (x + key * y) % w
                new_y = (x + (key + 1) * y) % h
                result[new_y, new_x] = temp[y, x]
    
    return result

def inverse_arnold_cat_map(img, iterations, key):
    """Apply inverse Arnold Cat Map to unscramble the image"""
    h, w = img.shape[0], img.shape[1]
    result = np.copy(img)
    
    for _ in range(iterations):
        temp = np.copy(result)
        for y in range(h):
            for x in range(w):
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
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        key = derive_key(key_str)
        key_int = int.from_bytes(key[:4], 'big') % 8 + 1
        
        scrambled = arnold_cat_map(img, iterations, key_int)
        img_bytes = pickle.dumps((scrambled.shape, scrambled.dtype, scrambled.tobytes()))
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted_bytes = iv + cipher.encrypt(pad(img_bytes, AES.block_size))
        
        metadata = {
            'shape': img.shape,
            'encrypted': True,
            'key_hash': hashlib.sha256(key_str.encode()).hexdigest()[:8],
            'iterations': iterations
        }
        
        meta_bytes = pickle.dumps(metadata)
        output = base64.b64encode(len(meta_bytes).to_bytes(4, 'big') + meta_bytes + encrypted_bytes)
        output_array = np.frombuffer(output, dtype=np.uint8)
        
        sqrt_size = int(np.ceil(np.sqrt(len(output_array))))
        padding = sqrt_size * sqrt_size - len(output_array)
        padded_array = np.pad(output_array, (0, padding), 'constant')
        output_img = padded_array.reshape(sqrt_size, sqrt_size)
        
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
        img_flat = img.flatten()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_flat = img[:,:,0].flatten()
        
        nonzero_indices = np.nonzero(img_flat)[0]
        if len(nonzero_indices) == 0:
            raise ValueError("Invalid encrypted image - no data found")
        
        last_nonzero = nonzero_indices[-1]
        img_flat = img_flat[:last_nonzero+1]
        
        try:
            decoded_data = base64.b64decode(img_flat)
        except Exception:
            raise ValueError("Not a valid encrypted image")
        
        meta_len = int.from_bytes(decoded_data[:4], 'big')
        metadata = pickle.loads(decoded_data[4:4+meta_len])
        
        if not metadata.get('encrypted', False):
            raise ValueError("Not an encrypted image")
        
        key = derive_key(key_str)
        if metadata.get('key_hash') != hashlib.sha256(key_str.encode()).hexdigest()[:8]:
            console.print("Warning: Key hash doesn't match. Decryption may fail.")
        
        encrypted_bytes = decoded_data[4+meta_len:]
        iv = encrypted_bytes[:16]
        ct = encrypted_bytes[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_bytes = unpad(cipher.decrypt(ct), AES.block_size)
        
        shape, dtype, img_bytes = pickle.loads(decrypted_bytes)
        scrambled = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
        key_int = int.from_bytes(key[:4], 'big') % 8 + 1
        unscrambled = inverse_arnold_cat_map(scrambled, iterations, key_int)
        
        return unscrambled
        
    except Exception as e:
        console.print(f"Decryption error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_image(input_path, output_path, key, iterations, mode):
    """Process images based on mode selection"""
    try:
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        if not any(input_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported file format. Please use: {', '.join(valid_extensions)}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' does not exist")
            
        file_size = os.path.getsize(input_path) / (1024 * 1024)
        if file_size > 100:
            raise ValueError(f"File too large ({file_size:.1f}MB). Maximum size is 100MB")
            
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except PermissionError:
                raise PermissionError(f"Unable to create output directory: {output_dir}. Check permissions.")
        
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image '{input_path}'. File may be corrupted or in an unsupported format.")
        
        max_dimension = 8192
        if img.shape[0] > max_dimension or img.shape[1] > max_dimension:
            raise ValueError(f"Image dimensions too large. Maximum allowed dimension is {max_dimension}px")

        if mode == "encrypt":
            if RICH_AVAILABLE:
                for step in track(range(3), description="Encrypting..."):
                    if step == 0:
                        time.sleep(0.1)
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
                        time.sleep(0.1)
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
        console.print(f"[bold red]Memory Error: Not enough memory to process this image.[/bold red]")
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
COLOR_PURPLE = "\033[95m"
COLOR_PINK = "\033[91m"
COLOR_BRIGHT_GREEN = "\033[92m"
COLOR_BRIGHT_YELLOW = "\033[93m"
COLOR_BRIGHT_BLUE = "\033[94m"
COLOR_BRIGHT_CYAN = "\033[96m"

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
    for i in range(len(banner)):
        sys.stdout.write(banner[i])
        sys.stdout.flush()
        time.sleep(0.005)
    print()
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
            "[bold cyan]║ [bold green][3] View Program Info                      [/bold green]║[/bold cyan]",
            "[bold cyan]║ [bold red][4] Exit                                   [/bold red]║[/bold cyan]",
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
        print("║ [3] View Program Info                      ║")
        print("║ [4] Exit                                   ║")
        print("╚════════════════════════════════════════════╝")

def display_info():
    info_text = """
About Pixel Encryption Tool

This tool provides a way to secure your images:

1. Encryption: Scrambles and encrypts images using:
   - Arnold Cat Map (pixel scrambling)
   - AES-256 encryption
   - Base64 encoding

Important Tips:
- Always remember your encryption key!
- When decrypting, use the same key and iterations used for encryption
- Larger images may take longer to process

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
    os.system('cls' if os.name == 'nt' else 'clear')
    display_banner()
    
    while True:
        draw_menu()
        choice = console.input("Select an option (1-4): ")
        
        if choice == "4":
            console.print("Exiting... Stay secure!")
            break
            
        if choice == "3":
            display_info()
            console.input("Press Enter to continue...")
            continue
        
        if choice in ["1", "2"]:
            print("="*40)
            
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
                
            while True:
                key = console.input("Enter encryption key: ")
                if not key or len(key) < 4:
                    console.print("Key must be at least 4 characters.")
                    continue
                break
                
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
            
            modes = {"1": "encrypt", "2": "decrypt"}
            console.print("Processing... Please wait...")
            
            success = process_image(input_path, output_path, key, iterations, modes[choice])
            
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
