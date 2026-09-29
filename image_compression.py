import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from collections import Counter
import heapq
import pickle
import os
from typing import Optional, cast


# ============================================================
# HUFFMAN NODE
# ============================================================

class Node:
    def __init__(
        self,
        value=None,
        frequency=0,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None
    ):
        self.value = value
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other: "Node") -> bool:
        return self.frequency < other.frequency


# ============================================================
# MAIN APPLICATION
# ============================================================

class CompressionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Image Compression: Huffman vs RLE")
        self.root.geometry("950x700")

        # Image variables
        self.image: Optional[Image.Image] = None
        self.pixels: list[int] = []
        self.image_path = ""

        # Huffman variables
        self.frequency = {}
        self.huffman_codes = {}
        self.huffman_tree: Optional[Node] = None
        self.huffman_bits = ""

        # RLE variables
        self.rle_data = []

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = tk.Label(
            root,
            text="Image Compression: Huffman vs RLE",
            font=("Arial", 20, "bold")
        )

        title.pack(pady=10)

        # ----------------------------------------------------
        # MAIN FRAME
        # ----------------------------------------------------

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # ----------------------------------------------------
        # LEFT FRAME
        # ----------------------------------------------------

        left_frame = tk.Frame(main_frame)
        left_frame.pack(
            side="left",
            fill="y",
            padx=15,
            pady=10
        )

        # ----------------------------------------------------
        # RIGHT FRAME
        # ----------------------------------------------------

        right_frame = tk.Frame(main_frame)
        right_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        buttons = [
            ("Upload Image", self.upload_image),
            ("Pixel Frequency", self.pixel_frequency),
            ("Build Huffman Codebook", self.build_huffman),
            ("Huffman Encode", self.huffman_encode),
            ("RLE Encode", self.rle_encode),
            ("Decode Huffman", self.decode_huffman),
            ("Decode RLE", self.decode_rle),
            ("Save Compressed Bitstream", self.save_file),
            ("Compare Compression", self.compare),
            ("Huffman Code Lengths", self.code_lengths)
        ]

        for text, command in buttons:

            btn = tk.Button(
                left_frame,
                text=text,
                command=command,
                width=28,
                height=2
            )

            btn.pack(pady=4)

        # ----------------------------------------------------
        # IMAGE AREA
        # ----------------------------------------------------

        self.image_label = tk.Label(
            right_frame,
            text="No image selected",
            font=("Arial", 14)
        )

        self.image_label.pack(pady=10)

        # ----------------------------------------------------
        # INFORMATION AREA
        # ----------------------------------------------------

        self.text = tk.Text(
            right_frame,
            width=65,
            height=25,
            font=("Consolas", 10)
        )

        self.text.pack(
            fill="both",
            expand=True
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = tk.Label(
            root,
            text="Ready - Upload an image",
            relief="sunken",
            anchor="w"
        )

        self.status.pack(
            side="bottom",
            fill="x"
        )

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    def show_text(self, data):

        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, data)

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    def upload_image(self):

        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files",
                 "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if path == "":
            return

        try:

            # Open image
            img = Image.open(path)

            # Convert to grayscale
            img = img.convert("L")

            self.image = img
            self.image_path = path

            # Convert image to list of pixels
            self.pixels = [int(pixel) for pixel in img.getdata()]  # type: ignore[assignment]

            # Reset old compression data
            self.frequency = {}
            self.huffman_codes = {}
            self.huffman_tree = None
            self.huffman_bits = ""
            self.rle_data = []

            # Display image
            preview = img.copy()
            preview.thumbnail((500, 400))

            self.photo = ImageTk.PhotoImage(preview)

            self.image_label.config(
                image=self.photo,
                text=""
            )

            width, height = img.size

            info = ""

            info += "IMAGE INFORMATION\n"
            info += "============================\n\n"

            info += "File: " + os.path.basename(path) + "\n"
            info += "Width: " + str(width) + " pixels\n"
            info += "Height: " + str(height) + " pixels\n"
            info += "Total Pixels: " + str(len(self.pixels)) + "\n"
            info += "Mode: " + img.mode + "\n"
            info += "Original File Size: " + str(
                os.path.getsize(path)
            ) + " bytes\n\n"

            info += "Image converted to grayscale successfully."

            self.show_text(info)

            self.status.config(
                text="Image uploaded successfully"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                "Could not open image.\n\n" + str(e)
            )

    # ========================================================
    # TASK 2 - PIXEL FREQUENCY
    # ========================================================

    def pixel_frequency(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        self.frequency = Counter(self.pixels)

        result = ""

        result += "PIXEL FREQUENCY\n"
        result += "============================\n\n"

        result += "Pixel Value     Frequency\n"
        result += "-------------------------\n"

        for value in sorted(self.frequency):

            result += (
                "{:<15}{}\n".format(
                    value,
                    self.frequency[value]
                )
            )

        result += "\n"
        result += "Unique pixel values: "
        result += str(len(self.frequency))

        self.show_text(result)

        # Optional histogram
        try:

            import matplotlib.pyplot as plt

            x = list(self.frequency.keys())
            y = list(self.frequency.values())

            plt.figure()

            plt.bar(x, y)

            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.title("Pixel Frequency")

            plt.show()

        except ImportError:

            pass

        self.status.config(
            text="Pixel frequency calculated"
        )

    # ========================================================
    # TASK 3 - BUILD HUFFMAN TREE
    # ========================================================

    def build_huffman(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        if len(self.frequency) == 0:

            self.frequency = Counter(self.pixels)

        heap = []

        # Create nodes
        for value in self.frequency:

            node = Node(
                value,
                self.frequency[value]
            )

            heapq.heappush(heap, node)

        # Only one unique pixel
        if len(heap) == 1:

            only = heapq.heappop(heap)

            root = Node(
                None,
                only.frequency,
                only,
                None
            )

            heapq.heappush(heap, root)

        # Build tree
        while len(heap) > 1:

            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            parent = Node(
                None,
                left.frequency + right.frequency,
                left,
                right
            )

            heapq.heappush(heap, parent)

        self.huffman_tree = heap[0]

        # Generate codes
        self.huffman_codes = {}

        self.generate_codes(
            self.huffman_tree,
            ""
        )

        result = ""

        result += "HUFFMAN CODEBOOK\n"
        result += "============================\n\n"

        result += "Pixel     Frequency     Code\n"
        result += "----------------------------\n"

        for value in sorted(self.frequency):

            result += (
                "{:<9}{:<14}{}\n".format(
                    value,
                    self.frequency[value],
                    self.huffman_codes[value]
                )
            )

        self.show_text(result)

        self.status.config(
            text="Huffman codebook created"
        )

    # ========================================================
    # GENERATE HUFFMAN CODES
    # ========================================================

    def generate_codes(self, node, code):

        if node is None:
            return

        # Leaf node
        if node.value is not None:

            if code == "":
                code = "0"

            self.huffman_codes[node.value] = code

            return

        self.generate_codes(
            node.left,
            code + "0"
        )

        self.generate_codes(
            node.right,
            code + "1"
        )

    # ========================================================
    # TASK 3 - HUFFMAN ENCODE
    # ========================================================

    def huffman_encode(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        if len(self.huffman_codes) == 0:

            self.build_huffman()

        self.huffman_bits = ""

        for pixel in self.pixels:

            self.huffman_bits += self.huffman_codes[pixel]

        original_bits = len(self.pixels) * 8

        compressed_bits = len(self.huffman_bits)

        ratio = original_bits / compressed_bits

        saved = (
            (1 - compressed_bits / original_bits)
            * 100
        )

        result = ""

        result += "HUFFMAN ENCODING\n"
        result += "============================\n\n"

        result += "Original size: "
        result += str(original_bits)
        result += " bits\n"

        result += "Compressed size: "
        result += str(compressed_bits)
        result += " bits\n\n"

        result += "Compression Ratio: "
        result += "{:.3f}:1\n".format(ratio)

        result += "Space Saved: "
        result += "{:.2f}%\n\n".format(saved)

        result += "Encoded Bitstream Preview:\n"
        result += "----------------------------\n"

        result += self.huffman_bits[:500]

        if len(self.huffman_bits) > 500:
            result += "..."

        self.show_text(result)

        self.status.config(
            text="Huffman encoding completed"
        )

    # ========================================================
    # TASK 4 - RLE ENCODING
    # ========================================================

    def rle_encode(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        self.rle_data = []

        current_pixel = self.pixels[0]
        count = 1

        for pixel in self.pixels[1:]:

            if pixel == current_pixel:

                count += 1

            else:

                self.rle_data.append(
                    (current_pixel, count)
                )

                current_pixel = pixel
                count = 1

        # Add final run
        self.rle_data.append(
            (current_pixel, count)
        )

        original_bits = len(self.pixels) * 8

        # Assume:
        # 8 bits for pixel
        # 32 bits for count
        rle_bits = len(self.rle_data) * 40

        ratio = original_bits / rle_bits

        saved = (
            (1 - rle_bits / original_bits)
            * 100
        )

        result = ""

        result += "RLE ENCODING\n"
        result += "============================\n\n"

        result += "Original pixels: "
        result += str(len(self.pixels))
        result += "\n"

        result += "Number of runs: "
        result += str(len(self.rle_data))
        result += "\n\n"

        result += "Original size: "
        result += str(original_bits)
        result += " bits\n"

        result += "Estimated RLE size: "
        result += str(rle_bits)
        result += " bits\n\n"

        result += "Compression Ratio: "
        result += "{:.3f}:1\n".format(ratio)

        result += "Space Saved: "
        result += "{:.2f}%\n\n".format(saved)

        result += "First 50 RLE pairs:\n"
        result += "----------------------------\n"

        for pair in self.rle_data[:50]:

            result += str(pair) + "\n"

        self.show_text(result)

        self.status.config(
            text="RLE encoding completed"
        )

    # ========================================================
    # TASK 5 - HUFFMAN DECODING
    # ========================================================

    def decode_huffman(self):

        if self.huffman_bits == "":

            messagebox.showwarning(
                "Warning",
                "Please perform Huffman encoding first."
            )

            return

        if self.image is None or self.huffman_tree is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image and build Huffman codebook first."
            )

            return

        try:

            decoded_pixels = []
            tree: Node = self.huffman_tree

            # Special case: only one pixel value
            left_node = tree.left
            if left_node is not None and left_node.value is not None:

                value = left_node.value

                decoded_pixels = [
                    value
                    for i in range(len(self.pixels))
                ]

            else:

                current: Node = tree

                for bit in self.huffman_bits:

                    if bit == "0":
                        next_node = current.left
                    else:
                        next_node = current.right

                    if next_node is None:
                        raise ValueError("Invalid Huffman bitstream")

                    current = next_node

                    if current.value is not None:

                        decoded_pixels.append(
                            current.value
                        )

                        current = tree

            # Create reconstructed image
            width, height = self.image.size

            decoded_image = Image.new(
                "L",
                (width, height)
            )

            decoded_image.putdata(
                decoded_pixels
            )

            # Check correctness
            correct = (
                decoded_pixels == self.pixels
            )

            decoded_image.show()

            result = ""

            result += "HUFFMAN DECODING\n"
            result += "============================\n\n"

            result += "Decoded pixels: "
            result += str(len(decoded_pixels))
            result += "\n"

            result += "Original pixels: "
            result += str(len(self.pixels))
            result += "\n\n"

            result += "Reconstruction successful: "
            result += str(correct)

            self.show_text(result)

            self.status.config(
                text="Huffman decoding completed"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                "Huffman decoding failed.\n\n" + str(e)
            )

    # ========================================================
    # TASK 5 - RLE DECODING
    # ========================================================

    def decode_rle(self):

        if len(self.rle_data) == 0:

            messagebox.showwarning(
                "Warning",
                "Please perform RLE encoding first."
            )

            return

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        try:

            decoded_pixels = []

            for pixel, count in self.rle_data:

                for i in range(count):

                    decoded_pixels.append(pixel)

            width, height = self.image.size

            decoded_image = Image.new(
                "L",
                (width, height)
            )

            decoded_image.putdata(
                decoded_pixels
            )

            correct = (
                decoded_pixels == self.pixels
            )

            decoded_image.show()

            result = ""

            result += "RLE DECODING\n"
            result += "============================\n\n"

            result += "Decoded pixels: "
            result += str(len(decoded_pixels))
            result += "\n"

            result += "Original pixels: "
            result += str(len(self.pixels))
            result += "\n\n"

            result += "Reconstruction successful: "
            result += str(correct)

            self.show_text(result)

            self.status.config(
                text="RLE decoding completed"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                "RLE decoding failed.\n\n" + str(e)
            )

    # ========================================================
    # TASK 6 - SAVE COMPRESSED BITSTREAM
    # ========================================================

    def save_file(self):

        if self.huffman_bits == "":

            messagebox.showwarning(
                "Warning",
                "Please perform Huffman encoding first."
            )

            return

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Save Compressed Bitstream",
            defaultextension=".bin",
            filetypes=[
                ("Binary File", "*.bin"),
                ("All Files", "*.*")
            ]
        )

        if path == "":
            return

        try:

            data = {
                "width": self.image.size[0],
                "height": self.image.size[1],
                "codes": self.huffman_codes,
                "bits": self.huffman_bits
            }

            with open(path, "wb") as file:

                pickle.dump(
                    data,
                    file
                )

            messagebox.showinfo(
                "Success",
                "Compressed bitstream saved successfully."
            )

            self.status.config(
                text="Compressed bitstream saved"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                "Could not save file.\n\n" + str(e)
            )

    # ========================================================
    # TASK 7 - COMPARE HUFFMAN AND RLE
    # ========================================================

    def compare(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        # Make sure both methods have been performed
        if self.huffman_bits == "":

            self.huffman_encode()

        if len(self.rle_data) == 0:

            self.rle_encode()

        original_bits = len(self.pixels) * 8

        huffman_bits = len(self.huffman_bits)

        rle_bits = len(self.rle_data) * 40

        huffman_ratio = (
            original_bits / huffman_bits
        )

        rle_ratio = (
            original_bits / rle_bits
        )

        huffman_saved = (
            1 - huffman_bits / original_bits
        ) * 100

        rle_saved = (
            1 - rle_bits / original_bits
        ) * 100

        if huffman_ratio > rle_ratio:

            better = "HUFFMAN"

        else:

            better = "RLE"

        result = ""

        result += "COMPRESSION COMPARISON\n"
        result += "============================\n\n"

        result += "Original size: "
        result += str(original_bits)
        result += " bits\n\n"

        result += "HUFFMAN\n"
        result += "----------------------------\n"

        result += "Compressed: "
        result += str(huffman_bits)
        result += " bits\n"

        result += "Ratio: "
        result += "{:.3f}:1\n".format(
            huffman_ratio
        )

        result += "Space saved: "
        result += "{:.2f}%\n\n".format(
            huffman_saved
        )

        result += "RLE\n"
        result += "----------------------------\n"

        result += "Number of runs: "
        result += str(len(self.rle_data))
        result += "\n"

        result += "Estimated size: "
        result += str(rle_bits)
        result += " bits\n"

        result += "Ratio: "
        result += "{:.3f}:1\n".format(
            rle_ratio
        )

        result += "Space saved: "
        result += "{:.2f}%\n\n".format(
            rle_saved
        )

        result += "RESULT\n"
        result += "============================\n"

        result += "Better compression: "
        result += better

        self.show_text(result)

        # Optional graph
        try:

            import matplotlib.pyplot as plt

            methods = [
                "Original",
                "Huffman",
                "RLE"
            ]

            sizes = [
                original_bits,
                huffman_bits,
                rle_bits
            ]

            plt.figure()

            plt.bar(
                methods,
                sizes
            )

            plt.xlabel("Method")
            plt.ylabel("Size (bits)")
            plt.title("Compression Comparison")

            plt.show()

        except ImportError:

            pass

        self.status.config(
            text="Compression comparison completed"
        )

    # ========================================================
    # TASK 8 - HUFFMAN CODE LENGTHS
    # ========================================================

    def code_lengths(self):

        if len(self.huffman_codes) == 0:

            messagebox.showwarning(
                "Warning",
                "Please build Huffman codebook first."
            )

            return

        result = ""

        result += "HUFFMAN CODE LENGTHS\n"
        result += "============================\n\n"

        result += "Pixel     Code             Length\n"
        result += "---------------------------------\n"

        for value in sorted(self.huffman_codes):

            code = self.huffman_codes[value]

            result += (
                "{:<10}{:<17}{}\n".format(
                    value,
                    code,
                    len(code)
                )
            )

        self.show_text(result)

        # Optional graph
        try:

            import matplotlib.pyplot as plt

            values = list(
                self.huffman_codes.keys()
            )

            lengths = []

            for value in values:

                lengths.append(
                    len(self.huffman_codes[value])
                )

            plt.figure()

            plt.bar(
                [str(x) for x in values],
                lengths
            )

            plt.xlabel("Pixel Value")
            plt.ylabel("Code Length")
            plt.title("Huffman Code Lengths")

            plt.show()

        except ImportError:

            pass

        self.status.config(
            text="Huffman code lengths displayed"
        )


# ============================================================
# START PROGRAM
# ============================================================

root = tk.Tk()

app = CompressionApp(root)

root.mainloop()