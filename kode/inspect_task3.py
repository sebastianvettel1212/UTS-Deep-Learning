import zipfile
import ast
import struct

def read_npy_header(raw_bytes):
    magic = raw_bytes[:6]
    assert magic == b'\x93NUMPY'
    major = raw_bytes[6]
    if major == 1:
        hlen = int.from_bytes(raw_bytes[8:10], 'little')
        hdr_str = raw_bytes[10:10+hlen].decode('latin1')
        offset = 10 + hlen
    else:
        hlen = int.from_bytes(raw_bytes[8:12], 'little')
        hdr_str = raw_bytes[12:12+hlen].decode('latin1')
        offset = 12 + hlen
    return ast.literal_eval(hdr_str.strip()), offset

with zipfile.ZipFile('data/clustering.npz') as z:
    with z.open('spektrum.npy') as f:
        hdr, offset = read_npy_header(f.read(1024))
        print("Clustering Spektrum shape:", hdr['shape'])
    
    with z.open('spektrum.npy') as f:
        f.read(offset)
        # Inspect first 5 spectra (each 128 float16 = 256 bytes)
        for i in range(5):
            raw = f.read(256)
            vals = struct.unpack('<128e', raw)
            min_v, max_v = min(vals), max(vals)
            has_nan = any(x != x for x in vals)
            print(f"  Spectrum {i}: min={min_v:.3f}, max={max_v:.3f}, has_nan={has_nan}, first_5={vals[:5]}")

    with z.open('gambar.npy') as f:
        hdr, offset = read_npy_header(f.read(1024))
        print("Clustering Gambar shape:", hdr['shape'])
