
class FileExtensionError(Exception):
    """ Only png files are accepted """
    pass

class FileSizeError(Exception):
    """ Input file is too large """
    pass

class ColorTypeError(Exception):
    """ This program can only parse png files with colorType equal to 2 or 6  """
    pass

class MetadataError(Exception):
    """ Is thrown, when there was an issue reading the metadata (most likely the user supplied the wrong file to be extracted) """
    pass
