import argparse
from argparse import ArgumentParser    
from PIL import Image, ImageFile
import os
import numpy as np
from errors import *

""" 1. Parsing is very fragile. Find methods in Image that'll parse metadata reliably. The struct library might be helpful.
    2. Consider storing metadata in a tEXT or zTXt chunk.
    3. Consider dividing this code into multiple source files.
    4. Read about &= and |= in numpy.
"""

def is_png(file_name):
    return file_name.split('.')[-1].lower() == "png"

def get_color_type(file_name):
    with open(file_name, "rb") as f:
        lines = f.read()
        color_type = lines[25]
    if color_type != 2 and color_type != 6:
        raise ColorTypeError(f"This program is unable to handle colorType = {color_type}. Choose a file with colorType equal to 2 or 6.")
    return color_type


def read_image(file_name):
    im = Image.open(file_name)
    return np.asarray(im, dtype=np.uint8)


def read_metadata(file_name):
    i = -1
    buff = np.array([])
    with open(file_name, "rb") as f:
        lines = f.read()
        # If someone provides an incorrect file, it'll loop until index out of bounds. Try to mitigate that.
        while lines[i] != 46:  # this is a dot in ASCII
            buff = np.append(buff, lines[i])
            i += -1
            if i < -100:
                raise MetadataError(f"`{file_name}` doesn't contain the metadata required to extract the hidden image. Are you sure {file_name} is the correct file?")
        buff = np.flip(buff)
        buff = [chr(int(digit)) for digit in buff]
        metadata_len = int("".join(buff))
        
        metadata = lines[i - metadata_len : i]
        metadata = [chr(m) for m in metadata]
        metadata = "".join(metadata)
        metadata = metadata.split('.')
        metadata = [int(m) for m in metadata]
        return metadata
        

def write_metadata(file_name, color_type, width, height):
    color_type_ascii = str(color_type).encode("ASCII")
    width_ascii = str(width).encode("ASCII")
    height_ascii = str(height).encode("ASCII")
    separator = '.'.encode("ASCII")

    metadata_len = len(color_type_ascii) + len(width_ascii) + len(height_ascii) + 2 # for two dots: color_type.width.height
    metadata_len_ascii = str(metadata_len).encode("ASCII")
    
    # Recall that this'll be after the IEND chunk, hence won't be visible.
    # those 3 values will be separated by dots
    with open(file_name, "ab") as f:
        f.write(separator)
        f.write(color_type_ascii)
        f.write(separator)
        f.write(width_ascii)
        f.write(separator)
        f.write(height_ascii)
        f.write(separator)
        f.write(metadata_len_ascii)
        

def embed(carrier_file_name, embed_file_name):
    if not is_png(carrier_file_name) or not is_png(embed_file_name):
        raise FileExtensionError("This program can only works with .png files.")
    
    # read the carrier image and embedded image
    carrier_prototype = read_image(carrier_file_name)
    embed_prototype = read_image(embed_file_name) 
    
    # all this metadata will be appended to the IEND chunk
    color_type = get_color_type(embed_file_name)
    height, width, _ = embed_prototype.shape

    # I'll flatten them for simplicity
    embed_flat = embed_prototype.reshape(-1)
    carrier_flat = carrier_prototype.reshape(-1)

    # an image can only be hidden using the LSB technique iff the carrier image is at least 8 times bigger
    if len(embed_flat) * 8 > len(carrier_flat):
        max_size = int(len(carrier_flat) / 8)
        raise FileSizeError(f"Given carrier file `{carrier_file_name}`, you can embed at most {max_size} bytes of raw image data.")

    # turns each uint8 into an array of bits
    carrier_flat_bits = np.unpackbits(carrier_flat)
    embed_flat_bits = np.unpackbits(embed_flat)
    
    # we're reading indices: 7, 15, 23, ..., embed_flat_bits.shape[0] * 8
    # you have to multiply by 8 the end index, because, you're taking every 8th index
    carrier_flat_bits[7 : embed_flat_bits.shape[0] * 8 : 8] = embed_flat_bits
    carrier_flat = np.packbits(carrier_flat_bits)
    
    # unflatten
    pixels_final = carrier_flat.reshape(carrier_prototype.shape)
    
    # save
    im = Image.fromarray(pixels_final)
    im.save("secret.png")
    
    write_metadata("secret.png", color_type, width, height)


def extract(file_name):
    if not is_png(file_name):
        raise FileExtensionError("This program can only deal with .png files.")
    
    color_type, width, height = read_metadata(file_name)
    
    # if color_type = 2, then a pixel is (R, G, B), if it's 6, then (R, G, B, A)
    pixel_len = 3 if color_type == 2 else 4
    
    # read the secret image, flatten it and convert decimal to binary
    secret_flat = read_image(file_name).reshape(-1) 
    secret_flat_bits = np.unpackbits(secret_flat)

    # read the LSB from the secret image
    reveal_flat_bits = secret_flat_bits[7 : width * height * pixel_len * 8 * 8 : 8]
    reveal_flat = np.packbits(reveal_flat_bits)  # from binary to decimal
   
    # unflatten - revert to the original shape
    reveal = reveal_flat.reshape((height, width, pixel_len))
    
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
    args = parser.parse_args()

    if args.carrier_file_name:
        validate_file(args.carrier_file_name)

        if args.embed_file_name:
            validate_file(args.embed_file_name)
            embed(args.carrier_file_name, args.embed_file_name)
    
    elif args.extract_file_name:
        validate_file(args.extract_file_name)
        extract(args.extract_file_name)

