import zipfile
import ast
import csv

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

# Load metadata
mics = []
with open('data/clustering_meta.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        mics.append(row[2])

with zipfile.ZipFile('data/clustering.npz') as z:
    with z.open('gambar.npy') as f:
        hdr, offset = read_npy_header(f.read(1024))
    with z.open('gambar.npy') as f:
        f.read(offset)
        # Check first 500 images: compute mean brightness by microscope
        m1_means = []
        m2_means = []
        img_size = 64 * 64
        for i in range(500):
            raw = f.read(img_size)
            mean_val = sum(raw) / img_size
            if mics[i] == 'M1':
                m1_means.append(mean_val)
            else:
                m2_means.append(mean_val)
        print(f"M1 count={len(m1_means)}, mean pixel intensity={sum(m1_means)/len(m1_means):.2f}")
        print(f"M2 count={len(m2_means)}, mean pixel intensity={sum(m2_means)/len(m2_means):.2f}")
