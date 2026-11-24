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

# Let's inspect sensor.npy in klasifikasi_train.npz
with zipfile.ZipFile('data/klasifikasi_train.npz') as z:
    with z.open('sensor.npy') as f:
        data = f.read(1024)
        hdr, offset = read_npy_header(data)
        print("Train Sensor shape:", hdr['shape'])

# Let's check how many samples have missing sensor data
# Shape is (48000, 3, 128) float16
# In float16, 3 * 128 = 384 values = 768 bytes per sample
# Let's inspect first 100 samples
with zipfile.ZipFile('data/klasifikasi_train.npz') as z:
    with z.open('sensor.npy') as f:
        # Seek past header
        header_data = f.read(1024)
        hdr, offset = read_npy_header(header_data)
        # re-open to seek correctly
    with z.open('sensor.npy') as f:
        f.read(offset)
        # read chunks of 768 bytes
        nan_samples = 0
        all_nan_samples = 0
        total = 1000
        for _ in range(total):
            sample_bytes = f.read(768)
            # unpack 384 half-precision floats (format '<384e')
            floats = struct.unpack('<384e', sample_bytes)
            # count NaNs (NaN != NaN)
            n_nan = sum(1 for x in floats if x != x)
            if n_nan > 0:
                nan_samples += 1
            if n_nan == 384:
                all_nan_samples += 1
        print(f"Out of first {total} samples:")
        print(f"  Samples with at least 1 NaN: {nan_samples} ({nan_samples/total*100:.1f}%)")
        print(f"  Samples with all NaN (completely missing sensor): {all_nan_samples} ({all_nan_samples/total*100:.1f}%)")
