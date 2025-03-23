import onnxruntime

# Load the ONNX model
model_path = "./ml_conformer_generator/ml_conformer_generator/weights/onnx/egnn_chembl_15_39.onnx"
session = onnxruntime.InferenceSession(model_path)

print("Model loaded successfully!")

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
output_type = session.get_outputs()[0].type

print(f"Input name: {input_name}, shape: {input_shape}, type: {input_type}")
print(f"Output name: {output_name}, shape: {output_shape}, type: {output_type}")