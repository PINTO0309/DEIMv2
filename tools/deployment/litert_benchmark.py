import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="saved_model/deimv2_hgnetv2_atto_wholebody34_170query_integer_quant.tflite")
signature = interpreter.get_signature_runner()

input_data = np.random.random_sample([1,3,192,192]).astype(np.float32)
output = signature(input_bgr=input_data)

N = 100
start = time.perf_counter()
for _ in range(N):
    output = signature(input_bgr=input_data)
end = time.perf_counter()

avg_time_ms = (end - start) / N * 1000
print(f"Average inference time: {avg_time_ms:.2f} ms")
