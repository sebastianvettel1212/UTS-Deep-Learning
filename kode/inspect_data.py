import zipfile
import ast

def read_npy_header(raw_bytes):
    magic = raw_bytes[:6]
    assert magic == b'\x93NUMPY'
    major = raw_bytes[6]
    if major == 1:
        hlen = int.from_bytes(raw_bytes[8:10], 'little')
        hdr_str = raw_bytes[10:10+hlen].decode('latin1')
    else:
        hlen = int.from_bytes(raw_bytes[8:12], 'little')
        hdr_str = raw_bytes[12:12+hlen].decode('latin1')
    return ast.literal_eval(hdr_str.strip())

for zf_name in ['klasifikasi_train.npz', 'klasifikasi_test.npz', 'clustering.npz']:
    with zipfile.ZipFile('data/' + zf_name) as z:
        print(f"=== {zf_name} ===")
        for name in z.namelist():
            with z.open(name) as f:
                header = read_npy_header(f.read(256))
                print(f"  {name}: shape={header['shape']}, dtype={header['descr']}")
