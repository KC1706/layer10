import os

def what(file, h=None):
    """
    Stub for Python 3.13+ missing imghdr module
    """
    if h is None:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
    
    if h.startswith(b'\211PNG\r\n\032\n'):
        return 'png'
    elif h.startswith(b'\377\330'):
        return 'jpeg'
    elif h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    return None
