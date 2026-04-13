import argparse
from argparse import ArgumentParser    
from PIL import Image, ImageFile
import os
import numpy as np

"""
1. Use exceptions instead of killing the process (read about exceptions as well).
2. Take a closer look at colorType. It's slightly wrong in this program.
3. I think naming of width and height may be confusing. Image array are likely the other way around. Code works though.
4. Try to put width and height as metadata in the secret.png. 
"""

def is_png(file_name):
    return file_name.split('.')[-1] == "png"

def get_color_type(pixel_array):
    """Checks the color type of a PNG. It's crucial for knowing what each pixel is made of.
    Currently it's very crude. Ideally it should read a specific byte in IHDR chunk."""
    color_type = pixel_array.shape[2]
    if color_type != 3 and color_type != 4:
        print(f"Error: this program is unable to handle colorType = {color_type}.")
        print("Use colorType = 3 or 4.")
        exit()
    return color_type


def read_image(file_name):
    im = Image.open(file_name)
    return np.asarray(im, dtype=np.uint8)

def embed(carrier_file_name, embed_file_name):
    if not is_png(carrier_file_name) or not is_png(embed_file_name):
        print("Error: This program can only works with .png files.")
        exit()
    
    # read the carrier image and embedded image
    carrier_prototype = read_image(carrier_file_name)
    embed_prototype = read_image(embed_file_name) 
    # colot_type is crucial, when extracting the embedded image
    color_type = get_color_type(embed_prototype)

    # I'll flatten them to make looping easier
    embed_flat = embed_prototype.reshape(-1)
    carrier_flat = carrier_prototype.reshape(-1)
    

    # image can only be hidden using the LSB technique iff the carrier image is at least 8 times bigger
    if len(embed_flat) * 8 > len(carrier_flat):
        max_size = len(carrier_flat) / (8 * 10**6)
        
        # pretty display purposes 
        unit = "MB" if max_size > 0.1 else "kB"
        max_size = max_size if unit == "MB" else max_size * 10**3
        print(f"Given carrier file `{carrier_file_name}`, you can at most embed an image of size {max_size} {unit}.")
        exit()

    print(f"Embedding a secret image of size {embed_prototype.shape[0]}x{embed_prototype.shape[1]}...")
    
    # turns each uint8 into an array of bits
    carrier_flat_bits = np.unpackbits(carrier_flat)
    embed_flat_bits = np.unpackbits(embed_flat)
    
    # 7, 15, 23, ..., embed_flat_bits.shape[0] * 8
    # you have to multiply by 8, because, you're taking every 8th index
    carrier_flat_bits[7 : embed_flat_bits.shape[0] * 8 : 8] = embed_flat_bits
    carrier_flat = np.packbits(carrier_flat_bits)
    
    # unflatten
    pixels_final = carrier_flat.reshape(carrier_prototype.shape)
    
    # save
    im = Image.fromarray(pixels_final) 
    im.save("secret.png")

    # Append to the end of the image color_type of the embedded image, to make extracting possible.
    # Recall that this'll be after the IEND chunk, hence won't be visible.
    with open("secret.png", "ab") as f:
        f.write(str(color_type).encode('ASCII'))


def extract(file_name, width, height):
    if not is_png(file_name):
        print("Error: This program can only deal with .png files.")
        exit()
    
    # read the colorType
    with open(file_name, "rb") as f:
        lines = f.readlines()
        color_type = int(chr(lines[-1][-1]))  # read the last byte
   
    # read the secret image, flatten it and convert decimal to binary
    secret_flat = read_image(file_name).reshape(-1) 
    secret_flat_bits = np.unpackbits(secret_flat)

    # read the LSB from the secret image
    reveal_flat_bits = secret_flat_bits[7 : width * height * color_type * 8 * 8 : 8]
    reveal_flat = np.packbits(reveal_flat_bits)  # from binary to decimal
   
    # unflatten - revert to the original shape
    reveal = reveal_flat.reshape((width, height, color_type))
    
    # save
    im = Image.fromarray(reveal)
    im.save("reveal.png")


def validate_file(file_name):
    if not os.path.exists(file_name):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError(f"{file_name} does not exist")
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='stegtool', usage='%(prog)s [options]')
    parser.add_argument("-f", "--file", dest="carrier_file_name", type=validate_file, help="carrier file")
    parser.add_argument("-e", "--embed", dest="embed_file_name", help="image to be embedded")
    parser.add_argument("-x", "--extract", dest="extract_file_name", type=validate_file, help="extract an image")
    parser.add_argument("-w", "--width", type=int, help="width of the secret image")
    parser.add_argument("-he", "--height", type=int, help="height of the secret image")
    args = parser.parse_args()

    
    if args.carrier_file_name:
        validate_file(args.carrier_file_name)

        if args.embed_file_name:
            validate_file(args.embed_file_name)
            embed(args.carrier_file_name, args.embed_file_name)
    
    elif args.extract_file_name:
        validate_file(args.extract_file_name)
        extract(args.extract_file_name, args.width, args.height)

