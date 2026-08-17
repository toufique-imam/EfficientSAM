
import os, sys, resource, numpy as np, coremltools as ct
os.chdir('/Users/nuhash/Desktop/other-codes/EfficientSAM'); sys.path.insert(0, '/Users/nuhash/Desktop/other-codes/EfficientSAM')
grid = int(sys.argv[1])
batch = 8
dec = ct.models.MLModel('weights/coreml/efficient_sam_vitt_decoder.mlpackage')
embed = np.random.rand(*(1, 256, 64, 64)).astype(np.float32)
n = grid * grid
for s in range(0, n, batch):
    dec.predict({
        "image_embeddings": embed,
        "point_coords": (np.random.rand(1, batch, 1, 2) * 1024).astype(np.float32),
        "point_labels": np.ones((1, batch, 1), np.float32),
    })
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
