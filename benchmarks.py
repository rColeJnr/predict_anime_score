import time
import numpy as np
import pickle
import onnxruntime as rt
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import xgboost as xgb

dummy_input = np.random.rand(1, 44).astype(np.float32)

with open('anime_data_model.pkl', 'rb') as f:
    model_pkl = pickle.load(f)

if hasattr(model_pkl, "get_booster"):
    booster = model_pkl.get_booster()
else:
    booster = model_pkl

booster.feature_names = None
booster.feature_types = None

initial_types = [('float_input', FloatTensorType([None, 44]))]

onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_types)

with open("anime_predictor.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("anime_predictor.onnx 👍🏾")

# --- BENCHMARK: PICKLE/XGBOOST ---
start = time.time()
for _ in range(1000):
    _ = model_pkl.predict(dummy_input)
pkl_time = (time.time() - start) / 1000
print(f"Pickle Inference: {pkl_time*1000:.4f} ms per row") # 1.8257 ms

# --- BENCHMARK: ONNX ---
sess = rt.InferenceSession("anime_predictor.onnx")
input_name = sess.get_inputs()[0].name

start = time.time()
for _ in range(1000):
    _ = sess.run(None, {input_name: dummy_input})
onnx_time = (time.time() - start) / 1000
print(f"ONNX Inference: {onnx_time*1000:.4f} ms per row") # 1.4659 ms\

# ONNX is slightly faster 👍🏾