import os

def what(file, h=None):
    f = None
    try:
        if h is None:
            if isinstance(file, (str, os.PathLike)):
                f = open(file, 'rb')
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
                f = file
                
        if h.startswith(b'\xff\xd8'):
            return 'jpeg'
        if h.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        if h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
            return 'gif'
        if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
            return 'webp'
        return None
    finally:
        if f and f is not file:
            f.close()
