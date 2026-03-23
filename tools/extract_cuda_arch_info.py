import json
import nvidia_arch as na

def extract_info():
    cuda_info = na.detect_ctk()
    cuda_info['architectures'] = na.get_architectures()
    version = cuda_info.get("version", "unknown")
    filename = f"info-cuda-{version}.json"
    
    print("\nExtracted CUDA and architecture information: \n")
    print(json.dumps(cuda_info, indent=2))
    print("\nSaving information to file...")
    
    with open(filename, "w") as f:
        json.dump(cuda_info, f, indent=2)
        print(f"Saved CUDA architecture info to {filename}")

if __name__ == "__main__":
    extract_info()
