import argparse
from argparse import ArgumentParser    
from PIL import Image, ImageFile
import os
import numpy as np

"""
add the right error message, when embedded message too large
"""

def set_lsb(value):
    return value | np.uint8(0b00000001)

def clear_lsb(value):
    return value & np.uint8(0b11111110) 

def set_bit(value, bit):
    return value | (np.uint8(0b00000001) << bit) 

def _flatten(values):
    if isinstance(values, np.ndarray):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)

def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))

def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.prod(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset

def unflatten(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result

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
    embed_flat = flatten(embed_prototype)
    carrier_flat = flatten(carrier_prototype)
    
    # image can only be hidden using the LSB technique iff the carrier image is at least 8 times bigger
    if len(embed_flat) * 8 > len(carrier_flat):
        max_size = len(carrier_flat) / (8 * 10**6)
        
        # pretty display purposes 
        unit = "MB" if max_size > 0.1 else "kB"
        max_size = max_size if unit == "MB" else max_size * 10**3
        print(f"Given carrier file `{carrier_file_name}`, you can at most embed an image of size {max_size} {unit}.")
        exit()

    print(f"Embedding a secret image of size {embed_prototype.shape[0]}x{embed_prototype.shape[1]}...")
    
    mask = np.uint8(0b10000000)  # used for extracting each bit
    i = 0  # points to the rgb value in a carrier image (pixels that consist of rgb values are also flattened)
    for rgb_val_embed in embed_flat: # go through all rgb values 
        for j in range(8):  # go through all bits in an rgb value in a pixel in an embedded image
            bit_rgb_val_embed = rgb_val_embed & (mask >> j)  # get the (7 - j)th bit of rgb value in pixel in embedded image
            if bit_rgb_val_embed:
                carrier_flat[i] = set_lsb(carrier_flat[i])
            else:
                 carrier_flat[i] = clear_lsb(carrier_flat[i])
            i += 1
    
    # unflatten
    pixels_final = unflatten(carrier_flat, carrier_prototype)
     
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

    # image containing the secret
    secret_prototype = read_image(file_name) 
    secret_flat = flatten(secret_prototype)
    
    with open(file_name, "rb") as f:
        lines = f.readlines()
        color_type = int(chr(lines[-1][-1]))  # read the last byte
    
    # this array will hold rgb values of the secret image
    reveal_flat = np.zeros(width * height * color_type, dtype=np.uint8)
    
    mask = np.uint8(0b00000001)  # used to extract the LSB
    i = 0  # points to rgb in the image we're recovering from
    for k in range(width * height * color_type):  # here we're writing to reveal_flat
        for j in range(8):  # assemble each rgb value bit by bit
            if (secret_flat[i] & mask) == 1: # our array is already all 0s, so we only flip bits to 1s when we have to  
                reveal_flat[k] = set_bit(reveal_flat[k], 7 - j)
            i += 1
   
   # unflatten
    reveal_prototype = np.zeros((width, height, color_type))
    reveal = unflatten(reveal_flat, reveal_prototype)
    
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

