#!/usr/bin/env python
"""
Test script to verify RTX 3090 setup for Unsloth training
"""

import torch
import sys
import os

def test_cuda_setup():
    """Test CUDA availability and GPU properties"""
    print("=" * 60)
    print("CUDA Setup Test")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
            # Check if it's an RTX 3090
            if props.major == 8 and props.minor == 6:
                print("  ✓ RTX 3090 detected (SM_86)")
            else:
                print(f"  ⚠ Warning: Expected SM_86 for RTX 3090, got SM_{props.major}{props.minor}")
    else:
        print("❌ CUDA not available!")
        return False
    
    return True

def test_memory_allocation():
    """Test memory allocation patterns"""
    print("\n" + "=" * 60)
    print("Memory Allocation Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Skipping - CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    
    # Test different tensor sizes
    test_sizes = [
        (1024, 1024),      # 4 MB
        (4096, 4096),      # 64 MB
        (8192, 8192),      # 256 MB
        (16384, 16384),    # 1 GB
    ]
    
    for size in test_sizes:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Allocate tensor
            tensor = torch.randn(size, device=device, dtype=torch.float32)
            allocated = tensor.element_size() * tensor.nelement() / 1024**3
            
            # Get memory stats
            allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
            reserved_mem = torch.cuda.memory_reserved(device) / 1024**3
            
            print(f"✓ Allocated {size[0]}x{size[1]} tensor ({allocated:.2f} GB)")
            print(f"  Memory allocated: {allocated_mem:.2f} GB")
            print(f"  Memory reserved: {reserved_mem:.2f} GB")
            
            del tensor
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM with {size[0]}x{size[1]} tensor")
            return False
    
    return True

def test_imports():
    """Test required package imports"""
    print("\n" + "=" * 60)
    print("Package Import Test")
    print("=" * 60)
    
    packages = [
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("bitsandbytes", "BitsAndBytes"),
        ("unsloth", "Unsloth"),
    ]
    
    all_imported = True
    
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: Import failed - {e}")
            all_imported = False
    
    # Special test for Flash Attention
    try:
        import flash_attn
        print(f"✓ Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("⚠ Flash Attention: Not installed (optional but recommended)")
    
    return all_imported

def test_small_model():
    """Test loading a small model"""
    print("\n" + "=" * 60)
    print("Small Model Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Skipping - CUDA not available")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "gpt2"  # Small model for testing
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        
        # Test inference
        inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Model loaded and inference successful")
        print(f"  Generated: {result}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("RTX 3090 Unsloth Setup Verification")
    print("=" * 60)
    
    tests = [
        ("CUDA Setup", test_cuda_setup),
        ("Memory Allocation", test_memory_allocation),
        ("Package Imports", test_imports),
        ("Small Model", test_small_model),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Your RTX 3090 setup is ready for Unsloth training.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
