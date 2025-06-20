import numpy as np
import onnxruntime as ort
import torch

def run_onnx_inference(onnx_path, batch_size=4, height=640, width=640):
    # Create ONNX runtime session
    sess = ort.InferenceSession(onnx_path)
    
    # Get input names and shapes
    input_details = sess.get_inputs()
    print("Model Input Details:")
    for inp in input_details:
        print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
    
    # Generate random input data based on actual model inputs
    input_dict = {}
    for inp in input_details:
        if inp.name == 'images':
            # Create image input
            input_dict[inp.name] = np.random.randn(batch_size, 3, height, width).astype(np.float32)
        else:
            # Create other inputs with appropriate shapes/types
            shape = [batch_size if dim == 'N' else dim for dim in inp.shape]
            shape = [dim if isinstance(dim, int) else 1 for dim in shape]
            
            # Use default types based on common patterns
            if inp.type == 'tensor(int64)':
                data = np.zeros(shape, dtype=np.int64)
            elif inp.type == 'tensor(float)':
                data = np.zeros(shape, dtype=np.float32)
            else:
                data = np.zeros(shape, dtype=np.float32)
                
            input_dict[inp.name] = data
            print(f"Generated input '{inp.name}': shape={shape}, dtype={data.dtype}")

    # Run inference
    outputs = sess.run(None, input_dict)
    
    # Print output details
    output_details = sess.get_outputs()
    print("\nModel Output Details:")
    for out, result in zip(output_details, outputs):
        print(f"Name: {out.name}, Shape: {result.shape}, Type: {out.type}")
        if result.size > 0:
            print(f"First 2 values: {result.flatten()[:2]}")
        else:
            print(f"Empty output")
    
    return outputs

if __name__ == "__main__":
    onnx_model_path = "/home/nery/RT-DETR/mask_rtdetr_pytorch/rtdetr_mask_swin_b_batch.onnx"
    print("Running inference with generated inputs...")
    outputs = run_onnx_inference(onnx_model_path)
