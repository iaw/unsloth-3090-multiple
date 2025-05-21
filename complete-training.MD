docker run --rm -it \
--gpus all \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_DEBUG=INFO \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e CUDA_LAUNCH_BLOCKING=1 \
-e TORCH_USE_CUDA_DSA=1 \
-e CUDA_VISIBLE_DEVICES=0,1,2 \
-v "$HOME/Desktop/Model Launchers/unsloth":/app/unsloth_run \
-v "/media/rgilbreth/MISC - M.2 4TB1/Models/Text/Qwen3-32B":/app/models/Qwen3-32B \
-v "/media/rgilbreth/88BE1A52BE1A38E0/Users/rober/OneDrive/Desktop/Reforger Data Training/RAG Parsing Script/combined-alpaca-prompts/Final":/app/data/unsloth_datasets \
-v "$HOME/Desktop/Docker-Unsloth-Output":/app/output \
-w /app/unsloth_run \
rgilbreth/rtx-5xxx-unsloth-training:rtx-5xxx-unsloth-training \
/bin/bash -c "accelerate launch --multi_gpu --num_processes=3 unsloth_Accelerate-Docker.py --output_dir /app/output"


==========
== CUDA ==
==========

CUDA Version 12.8.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

The following values were not passed to `accelerate launch` and had defaults used instead:
	`--num_machines` was set to a value of `1`
	`--mixed_precision` was set to a value of `'no'`
	`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
[PID 125] Patched HfFileSystem.glob for local paths.
[PID 125] Script start. Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
[PID 125] Current PWD: /app/unsloth_run
[PID 125] TORCH_DISTRIBUTED_USE_DTENSOR: 0
[PID 125] CUDA_VISIBLE_DEVICES (from env): 0,1,2
[PID 125] ACCELERATE_USE_TP: false
[PID 125] Launcher Env: RANK=0, LOCAL_RANK=0, WORLD_SIZE=3
[PID 126] Patched HfFileSystem.glob for local paths.
[PID 126] Script start. Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
[PID 126] Current PWD: /app/unsloth_run
[PID 126] TORCH_DISTRIBUTED_USE_DTENSOR: 0
[PID 126] CUDA_VISIBLE_DEVICES (from env): 0,1,2
[PID 126] ACCELERATE_USE_TP: false
[PID 126] Launcher Env: RANK=1, LOCAL_RANK=1, WORLD_SIZE=3
[PID 127] Patched HfFileSystem.glob for local paths.
[PID 127] Script start. Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
[PID 127] Current PWD: /app/unsloth_run
[PID 127] TORCH_DISTRIBUTED_USE_DTENSOR: 0
[PID 127] CUDA_VISIBLE_DEVICES (from env): 0,1,2
[PID 127] ACCELERATE_USE_TP: false
[PID 127] Launcher Env: RANK=2, LOCAL_RANK=2, WORLD_SIZE=3
[PID 127, Rank 2] Imported torch. Version: 2.8.0a0+git5506baa. CUDA available: True
[PID 126, Rank 1] Imported torch. Version: 2.8.0a0+git5506baa. CUDA available: True
[PID 125, Rank 0] Imported torch. Version: 2.8.0a0+git5506baa. CUDA available: True
[PID 127, Rank 2] CUDA device count: 3
[PID 126, Rank 1] CUDA device count: 3
[PID 127, Rank 2] Current CUDA device (by torch.cuda.current_device()): 0
[PID 127, Rank 2] Name of current CUDA device: NVIDIA GeForce RTX 5090
[PID 125, Rank 0] CUDA device count: 3
[PID 126, Rank 1] Current CUDA device (by torch.cuda.current_device()): 0
[PID 126, Rank 1] Name of current CUDA device: NVIDIA GeForce RTX 5090
[PID 125, Rank 0] Current CUDA device (by torch.cuda.current_device()): 0
[PID 125, Rank 0] Name of current CUDA device: NVIDIA GeForce RTX 5090
⚠️ [PID 127, Rank 2] Error during DTensor patching: hasattr expected 2 arguments, got 1
[PID 127, Rank 2] Importing accelerate...
⚠️ [PID 126, Rank 1] Error during DTensor patching: hasattr expected 2 arguments, got 1
[PID 126, Rank 1] Importing accelerate...
⚠️ [PID 125, Rank 0] Error during DTensor patching: hasattr expected 2 arguments, got 1
[PID 125, Rank 0] Importing accelerate...
[PID 127, Rank 2] Imported accelerate.
[PID 127, Rank 2] Importing Unsloth...
[PID 126, Rank 1] Imported accelerate.
[PID 126, Rank 1] Importing Unsloth...
[PID 125, Rank 0] Imported accelerate.
[PID 125, Rank 0] Importing Unsloth...
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
🦥 Unsloth Zoo will now patch everything to make training faster!
🦥 Unsloth Zoo will now patch everything to make training faster!
[PID 126, Rank 1] Imported Unsloth.
[PID 126, Rank 1] Importing Transformers & Datasets...
[PID 126, Rank 1] Imported Transformers & Datasets.
[PID 126] Script __main__ started.
[PID 126, Pre-Accelerator-Rank] In main(). Initializing Accelerator...
[PID 127, Rank 2] Imported Unsloth.
[PID 127, Rank 2] Importing Transformers & Datasets...
[PID 127, Rank 2] Imported Transformers & Datasets.
[PID 127] Script __main__ started.
[PID 127, Pre-Accelerator-Rank] In main(). Initializing Accelerator...
[PID 125, Rank 0] Imported Unsloth.
[PID 125, Rank 0] Importing Transformers & Datasets...
[PID 125, Rank 0] Imported Transformers & Datasets.
[PID 125] Script __main__ started.
[PID 125, Pre-Accelerator-Rank] In main(). Initializing Accelerator...
[Gloo] Rank 0 is connected to 2 peer ranks. Expected number of connected peer ranks is : 2
[Gloo] Rank 2[Gloo] Rank  is connected to 1 is connected to 22 peer ranks.  peer ranks. Expected number of connected peer ranks is : Expected number of connected peer ranks is : 22

[PID 125, Rank 0] Accelerator initialized. Distributed: DistributedType.MULTI_GPU, Device: cuda:0, Num_processes: 3
[PID 125, Rank 0] Loading model and tokenizer...
[PID 125, Rank 0] In load_model()...
[PID 125, Rank 0] Using device_map: {'': device(type='cuda', index=0)}
[PID 125, Rank 0] model_kwargs: {'model_name': '/app/models/Qwen3-32B', 'max_seq_length': 8192, 'load_in_4bit': True, 'attn_implementation': 'flash_attention_2', 'device_map': {'': device(type='cuda', index=0)}, 'dtype': torch.bfloat16}
[PID 125, Rank 0] Calling FastLanguageModel.from_pretrained...
[PID 126, Rank 1] Accelerator initialized. Distributed: DistributedType.MULTI_GPU, Device: cuda:1, Num_processes: 3
[PID 126, Rank 1] Loading model and tokenizer...
[PID 126, Rank 1] In load_model()...
[PID 126, Rank 1] Using device_map: {'': device(type='cuda', index=1)}
[PID 126, Rank 1] model_kwargs: {'model_name': '/app/models/Qwen3-32B', 'max_seq_length': 8192, 'load_in_4bit': True, 'attn_implementation': 'flash_attention_2', 'device_map': {'': device(type='cuda', index=1)}, 'dtype': torch.bfloat16}
[PID 126, Rank 1] Calling FastLanguageModel.from_pretrained...
[PID 127, Rank 2] Accelerator initialized. Distributed: DistributedType.MULTI_GPU, Device: cuda:2, Num_processes: 3
[PID 127, Rank 2] Loading model and tokenizer...
[PID 127, Rank 2] In load_model()...
[PID 127, Rank 2] Using device_map: {'': device(type='cuda', index=2)}
[PID 127, Rank 2] model_kwargs: {'model_name': '/app/models/Qwen3-32B', 'max_seq_length': 8192, 'load_in_4bit': True, 'attn_implementation': 'flash_attention_2', 'device_map': {'': device(type='cuda', index=2)}, 'dtype': torch.bfloat16}
[PID 127, Rank 2] Calling FastLanguageModel.from_pretrained...
==((====))==  Unsloth 2025.5.6: Fast Qwen3 patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 5090. Num GPUs = 3. Max memory: 31.354 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0a0+git5506baa. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
==((====))==  Unsloth 2025.5.6: Fast Qwen3 patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 5090. Num GPUs = 3. Max memory: 31.354 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0a0+git5506baa. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
==((====))==  Unsloth 2025.5.6: Fast Qwen3 patching. Transformers: 4.51.3.
   \\   /|    NVIDIA GeForce RTX 5090. Num GPUs = 3. Max memory: 31.354 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.8.0a0+git5506baa. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|██████████████████| 17/17 [00:35<00:00,  2.10s/it]:02,  2.22s/it]
Loading checkpoint shards: 100%|█████████████████████████████████| 17/17 [00:36<00:00,  2.12s/it]
[PID 127, Rank 2] FastLanguageModel.from_pretrained successful.
[PID 127, Rank 2] Model device after load: cuda:2
[PID 127, Rank 2] Model and tokenizer loaded.
[PID 127, Rank 2] Applying LoRA...
[PID 127, Rank 2] In apply_lora()...
[PID 125, Rank 0] FastLanguageModel.from_pretrained successful.
[PID 125, Rank 0] Model device after load: cuda:0
[PID 125, Rank 0] Model and tokenizer loaded.
[PID 125, Rank 0] Applying LoRA...
[PID 125, Rank 0] In apply_lora()...
Loading checkpoint shards: 100%|█████████████████████████████████| 17/17 [00:36<00:00,  2.17s/it]
Unsloth 2025.5.6 patched 64 layers with 64 QKV layers, 64 O layers and 64 MLP layers.
[PID 126, Rank 1] FastLanguageModel.from_pretrained successful.
[PID 126, Rank 1] Model device after load: cuda:1
[PID 126, Rank 1] Model and tokenizer loaded.
[PID 126, Rank 1] Applying LoRA...
[PID 126, Rank 1] In apply_lora()...
Unsloth 2025.5.6 patched 64 layers with 64 QKV layers, 64 O layers and 64 MLP layers.
[PID 127, Rank 2] apply_lora successful.
[PID 127, Rank 2] LoRA applied.
[PID 127, Rank 2] Loading and splitting dataset...
[PID 127, Rank 2] In load_and_split_dataset()...
Generating train split: 75987 examples [00:00, 374295.89 examples/s]
[PID 127, Rank 2] load_and_split_dataset successful.
[PID 127, Rank 2] Dataset loaded and split.
[PID 127, Rank 2] Getting chat template...
[PID 127, Rank 2] Chat template applied.
[PID 127, Rank 2] Passing through train_ds text as-is (num_proc=1)...
Map:   8%|████▌                                                        | 5000/66579 [00:00<00:01, 49711.80 examples/s][PID 125, Rank 0] apply_lora successful.
[PID 125, Rank 0] LoRA applied.
[PID 125, Rank 0] Loading and splitting dataset...
[PID 125, Rank 0] In load_and_split_dataset()...
Map:  16%|█████████▎                                                  | 10362/66579 [00:00<00:01, 52001.59 examples/s]Unsloth 2025.5.6 patched 64 layers with 64 QKV layers, 64 O layers and 64 MLP layers.
Map:  24%|██████████████▏                                             | 15763/66579 [00:00<00:00, 52761.84 examples/s][PID 125, Rank 0] load_and_split_dataset successful.
[PID 125, Rank 0] Dataset loaded and split.
[PID 125, Rank 0] Getting chat template...
[PID 125, Rank 0] Chat template applied.
[PID 125, Rank 0] Passing through train_ds text as-is (num_proc=1)...
Map:  46%|███████████████████████████▍                                | 30487/66579 [00:00<00:00, 51147.67 examples/s][PID 126, Rank 1] apply_lora successful.
[PID 126, Rank 1] LoRA applied.
[PID 126, Rank 1] Loading and splitting dataset...
[PID 126, Rank 1] In load_and_split_dataset()...
Map:  95%|█████████████████████████████████████████████████████████   | 63373/66579 [00:01<00:00, 52691.82 examples/s][PID 126, Rank 1] load_and_split_dataset successful.
[PID 126, Rank 1] Dataset loaded and split.
[PID 126, Rank 1] Getting chat template...
[PID 126, Rank 1] Chat template applied.
[PID 126, Rank 1] Passing through train_ds text as-is (num_proc=1)...
Map: 100%|████████████████████████████████████████████████████████████| 66579/66579 [00:01<00:00, 47001.72 examples/s]
[PID 127, Rank 2] Passing through val_ds text as-is (batched=True)...
Map: 100%|█████████████████████████████████████████████████████████████| 9408/9408 [00:00<00:00, 265252.43 examples/s]
[PID 127, Rank 2] Datasets processed.
Map:  88%|████████████████████████████████████████████████████▋       | 58433/66579 [00:01<00:00, 50798.10 examples/s][PID 127, Rank 2] TrainingArguments initialized.
[PID 127, Rank 2] Initializing SFTTrainer...
Map: 100%|████████████████████████████████████████████████████████████| 66579/66579 [00:01<00:00, 44406.72 examples/s]
[PID 125, Rank 0] Passing through val_ds text as-is (batched=True)...
[PID 125, Rank 0] Datasets processed.
[PID 125, Rank 0] TrainingArguments initialized.
[PID 125, Rank 0] Initializing SFTTrainer...
Map: 100%|████████████████████████████████████████████████████████████| 66579/66579 [00:02<00:00, 30444.03 examples/s]
[PID 126, Rank 1] Passing through val_ds text as-is (batched=True)...
[PID 126, Rank 1] Datasets processed.
[PID 126, Rank 1] TrainingArguments initialized.
[PID 126, Rank 1] Initializing SFTTrainer...
Unsloth: Tokenizing ["text"]: 100%|████████████████████████████████████| 66579/66579 [00:11<00:00, 5645.27 examples/s]
Unsloth: Tokenizing ["text"]: 100%|████████████████████████████████████| 66579/66579 [00:11<00:00, 5558.14 examples/s]
Unsloth: Tokenizing ["text"]: 100%|████████████████████████████████████| 66579/66579 [00:11<00:00, 6011.45 examples/s]
Unsloth: Tokenizing ["text"]: 100%|██████████████████████████████████████| 9408/9408 [00:01<00:00, 6362.49 examples/s]
[PID 127, Rank 2] SFTTrainer initialized. Model is on: cuda:2
Unsloth: Tokenizing ["text"]: 100%|██████████████████████████████████████| 9408/9408 [00:01<00:00, 8297.21 examples/s]
[PID 125, Rank 0] SFTTrainer initialized. Model is on: cuda:0
[PID 125, Rank 0] MAIN PROCESS: Pre-train check for model.config.use_cache.
✅ [PID 125, Rank 0] MAIN PROCESS: Forcing model.config.use_cache = False on unwrapped model.
Unsloth: Tokenizing ["text"]: 100%|██████████████████████████████████████| 9408/9408 [00:01<00:00, 9006.51 examples/s]
[PID 126, Rank 1] SFTTrainer initialized. Model is on: cuda:1
[rank0]:[W521 02:15:38.615073931 ProcessGroupNCCL.cpp:4872] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can specify device_id in init_process_group() to force use of a particular device.
[rank1]:[W521 02:15:38.615082433 ProcessGroupNCCL.cpp:4872] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can specify device_id in init_process_group() to force use of a particular device.
[rank2]:[W521 02:15:38.615104517 ProcessGroupNCCL.cpp:4872] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can specify device_id in init_process_group() to force use of a particular device.
7841b93df7f1:125:125 [0] NCCL INFO Bootstrap: Using eth0:172.17.0.2<0>
7841b93df7f1:125:125 [0] NCCL INFO cudaDriverVersion 12090
7841b93df7f1:126:126 [1] NCCL INFO cudaDriverVersion 12090
7841b93df7f1:127:127 [2] NCCL INFO cudaDriverVersion 12090
7841b93df7f1:125:125 [0] NCCL INFO NCCL version 2.26.5+cuda12.8
7841b93df7f1:125:125 [0] NCCL INFO Comm config Blocking set to 1
7841b93df7f1:126:126 [1] NCCL INFO Bootstrap: Using eth0:172.17.0.2<0>
7841b93df7f1:126:126 [1] NCCL INFO NCCL version 2.26.5+cuda12.8
7841b93df7f1:127:127 [2] NCCL INFO Bootstrap: Using eth0:172.17.0.2<0>
7841b93df7f1:127:127 [2] NCCL INFO NCCL version 2.26.5+cuda12.8
7841b93df7f1:126:126 [1] NCCL INFO Comm config Blocking set to 1
7841b93df7f1:127:127 [2] NCCL INFO Comm config Blocking set to 1
7841b93df7f1:127:589 [2] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal net plugin.
7841b93df7f1:127:589 [2] NCCL INFO Failed to open libibverbs.so[.1]
7841b93df7f1:127:589 [2] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
7841b93df7f1:127:589 [2] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
7841b93df7f1:127:589 [2] NCCL INFO Using network Socket
7841b93df7f1:126:588 [1] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal net plugin.
7841b93df7f1:126:588 [1] NCCL INFO Failed to open libibverbs.so[.1]
7841b93df7f1:126:588 [1] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
7841b93df7f1:126:588 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
7841b93df7f1:126:588 [1] NCCL INFO Using network Socket
7841b93df7f1:127:589 [2] NCCL INFO ncclCommInitRankConfig comm 0x370962c0 rank 2 nranks 3 cudaDev 2 nvmlDev 2 busId e1000 commId 0x8ce84ec70d264f5f - Init START
7841b93df7f1:126:588 [1] NCCL INFO ncclCommInitRankConfig comm 0x57451020 rank 1 nranks 3 cudaDev 1 nvmlDev 1 busId c2000 commId 0x8ce84ec70d264f5f - Init START
7841b93df7f1:125:587 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal net plugin.
7841b93df7f1:125:587 [0] NCCL INFO Failed to open libibverbs.so[.1]
7841b93df7f1:125:587 [0] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
7841b93df7f1:125:587 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
7841b93df7f1:125:587 [0] NCCL INFO Using network Socket
7841b93df7f1:125:587 [0] NCCL INFO ncclCommInitRankConfig comm 0x4e244b50 rank 0 nranks 3 cudaDev 0 nvmlDev 0 busId c1000 commId 0x8ce84ec70d264f5f - Init START
7841b93df7f1:127:589 [2] NCCL INFO RAS client listening socket at ::1<28028>
7841b93df7f1:126:588 [1] NCCL INFO RAS client listening socket at ::1<28028>
7841b93df7f1:125:587 [0] NCCL INFO RAS client listening socket at ::1<28028>
7841b93df7f1:127:589 [2] NCCL INFO Bootstrap timings total 0.002916 (create 0.000029, send 0.000088, recv 0.002392, ring 0.000131, delay 0.000000)
7841b93df7f1:125:587 [0] NCCL INFO Bootstrap timings total 0.000692 (create 0.000027, send 0.000082, recv 0.000271, ring 0.000038, delay 0.000001)
7841b93df7f1:126:588 [1] NCCL INFO Bootstrap timings total 0.002621 (create 0.000026, send 0.000072, recv 0.000082, ring 0.000039, delay 0.000000)
7841b93df7f1:126:588 [1] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
7841b93df7f1:126:588 [1] NCCL INFO NVLS multicast support is not available on dev 1
7841b93df7f1:125:587 [0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
7841b93df7f1:127:589 [2] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
7841b93df7f1:125:587 [0] NCCL INFO NVLS multicast support is not available on dev 0
7841b93df7f1:127:589 [2] NCCL INFO NVLS multicast support is not available on dev 2
7841b93df7f1:127:589 [2] NCCL INFO comm 0x370962c0 rank 2 nRanks 3 nNodes 1 localRanks 3 localRank 2 MNNVL 0
7841b93df7f1:126:588 [1] NCCL INFO comm 0x57451020 rank 1 nRanks 3 nNodes 1 localRanks 3 localRank 1 MNNVL 0
7841b93df7f1:125:587 [0] NCCL INFO comm 0x4e244b50 rank 0 nRanks 3 nNodes 1 localRanks 3 localRank 0 MNNVL 0
7841b93df7f1:127:589 [2] NCCL INFO Trees [0] -1/-1/-1->2->1 [1] -1/-1/-1->2->1 [2] -1/-1/-1->2->1 [3] -1/-1/-1->2->1
7841b93df7f1:126:588 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0
7841b93df7f1:125:587 [0] NCCL INFO Channel 00/04 : 0 1 2
7841b93df7f1:127:589 [2] NCCL INFO P2P Chunksize set to 131072
7841b93df7f1:126:588 [1] NCCL INFO P2P Chunksize set to 131072
7841b93df7f1:125:587 [0] NCCL INFO Channel 01/04 : 0 1 2
7841b93df7f1:125:587 [0] NCCL INFO Channel 02/04 : 0 1 2
7841b93df7f1:125:587 [0] NCCL INFO Channel 03/04 : 0 1 2
7841b93df7f1:125:587 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1
7841b93df7f1:125:587 [0] NCCL INFO P2P Chunksize set to 131072
7841b93df7f1:125:587 [0] NCCL INFO Check P2P Type intraNodeP2pSupport 0 directMode 0
7841b93df7f1:126:594 [1] NCCL INFO [Proxy Service] Device 1 CPU core 10
7841b93df7f1:127:593 [2] NCCL INFO [Proxy Service] Device 2 CPU core 1
7841b93df7f1:126:597 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 7
7841b93df7f1:125:595 [0] NCCL INFO [Proxy Service] Device 0 CPU core 42
7841b93df7f1:125:598 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 27
7841b93df7f1:127:596 [2] NCCL INFO [Proxy Service UDS] Device 2 CPU core 43
7841b93df7f1:127:589 [2] NCCL INFO threadThresholds 8/8/64 | 24/8/64 | 512 | 512
7841b93df7f1:127:589 [2] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
7841b93df7f1:126:588 [1] NCCL INFO threadThresholds 8/8/64 | 24/8/64 | 512 | 512
7841b93df7f1:126:588 [1] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
7841b93df7f1:125:587 [0] NCCL INFO threadThresholds 8/8/64 | 24/8/64 | 512 | 512
7841b93df7f1:125:587 [0] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
7841b93df7f1:125:587 [0] NCCL INFO CC Off, workFifoBytes 1048576
7841b93df7f1:126:588 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
7841b93df7f1:127:589 [2] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
7841b93df7f1:125:587 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
7841b93df7f1:126:588 [1] NCCL INFO ncclCommInitRankConfig comm 0x57451020 rank 1 nranks 3 cudaDev 1 nvmlDev 1 busId c2000 commId 0x8ce84ec70d264f5f - Init COMPLETE
7841b93df7f1:127:589 [2] NCCL INFO ncclCommInitRankConfig comm 0x370962c0 rank 2 nranks 3 cudaDev 2 nvmlDev 2 busId e1000 commId 0x8ce84ec70d264f5f - Init COMPLETE
7841b93df7f1:126:588 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 3 total 0.17 (kernels 0.15, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
7841b93df7f1:125:587 [0] NCCL INFO ncclCommInitRankConfig comm 0x4e244b50 rank 0 nranks 3 cudaDev 0 nvmlDev 0 busId c1000 commId 0x8ce84ec70d264f5f - Init COMPLETE
7841b93df7f1:127:589 [2] NCCL INFO Init timings - ncclCommInitRankConfig: rank 2 nranks 3 total 0.17 (kernels 0.15, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
7841b93df7f1:125:587 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 3 total 0.17 (kernels 0.16, alloc 0.00, bootstrap 0.00, allgathers 0.00, topo 0.01, graphs 0.00, connections 0.00, rest 0.00)
7841b93df7f1:126:600 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
7841b93df7f1:127:601 [2] NCCL INFO Channel 00 : 2[2] -> 0[0] via SHM/direct/direct
7841b93df7f1:125:599 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
7841b93df7f1:126:600 [1] NCCL INFO Channel 01 : 1[1] -> 2[2] via SHM/direct/direct
7841b93df7f1:125:599 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
7841b93df7f1:127:601 [2] NCCL INFO Channel 01 : 2[2] -> 0[0] via SHM/direct/direct
7841b93df7f1:126:600 [1] NCCL INFO Channel 02 : 1[1] -> 2[2] via SHM/direct/direct
7841b93df7f1:125:599 [0] NCCL INFO Channel 02 : 0[0] -> 1[1] via SHM/direct/direct
7841b93df7f1:127:601 [2] NCCL INFO Channel 02 : 2[2] -> 0[0] via SHM/direct/direct
7841b93df7f1:125:599 [0] NCCL INFO Channel 03 : 0[0] -> 1[1] via SHM/direct/direct
7841b93df7f1:126:600 [1] NCCL INFO Channel 03 : 1[1] -> 2[2] via SHM/direct/direct
7841b93df7f1:127:601 [2] NCCL INFO Channel 03 : 2[2] -> 0[0] via SHM/direct/direct
7841b93df7f1:126:600 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
7841b93df7f1:125:599 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
7841b93df7f1:127:601 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[PID 126, Rank 1] All processes ready. Calling trainer.train()...[PID 125, Rank 0] All processes ready. Calling trainer.train()...

[PID 127, Rank 2] All processes ready. Calling trainer.train()...
Unsloth is running with multi GPUs - the effective batch size is multiplied by 3
Unsloth is running with multi GPUs - the effective batch size is multiplied by 3
Unsloth is running with multi GPUs - the effective batch size is multiplied by 3
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 66,579 | Num Epochs = 1 | Total steps = 30
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 3 | Total batch size (1 x 1 x 3) = 3
 "-____-"     Trainable parameters = 134,217,728/17,293,530,112 (0.78% trained)
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 66,579 | Num Epochs = 1 | Total steps = 30
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 3 | Total batch size (1 x 1 x 3) = 3
 "-____-"     Trainable parameters = 134,217,728/17,293,530,112 (0.78% trained)
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 66,579 | Num Epochs = 1 | Total steps = 30
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 3 | Total batch size (1 x 1 x 3) = 3
 "-____-"     Trainable parameters = 134,217,728/17,293,530,112 (0.78% trained)
{'loss': 2.7964, 'grad_norm': 0.4271438419818878, 'learning_rate': 0.0, 'epoch': 0.0}                                 
  3%|██▋                                                                               | 1/30 [00:02<01:21,  2.82s/it]Unsloth: Will smartly offload gradients to save VRAM!
Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 2.9399, 'grad_norm': 0.42266279458999634, 'learning_rate': 4e-05, 'epoch': 0.0}                              
  7%|█████▍                                                                            | 2/30 [00:03<00:50,  1.81s/it]Unsloth: Will smartly offload gradients to save VRAM!
{'loss': 2.925, 'grad_norm': 0.4378184974193573, 'learning_rate': 8e-05, 'epoch': 0.0}                                
{'loss': 3.0178, 'grad_norm': 0.5515595078468323, 'learning_rate': 0.00012, 'epoch': 0.0}                             
{'loss': 2.7177, 'grad_norm': 0.7373011708259583, 'learning_rate': 0.00016, 'epoch': 0.0}                             
{'loss': 2.6057, 'grad_norm': 0.9584798812866211, 'learning_rate': 0.0002, 'epoch': 0.0}                              
{'loss': 2.2476, 'grad_norm': 0.8489506244659424, 'learning_rate': 0.000192, 'epoch': 0.0}                            
{'loss': 1.8753, 'grad_norm': 1.084259271621704, 'learning_rate': 0.00018400000000000003, 'epoch': 0.0}               
{'loss': 1.5157, 'grad_norm': 0.8525004982948303, 'learning_rate': 0.00017600000000000002, 'epoch': 0.0}              
{'loss': 1.4367, 'grad_norm': 1.420272946357727, 'learning_rate': 0.000168, 'epoch': 0.0}                             
{'loss': 1.2244, 'grad_norm': 0.7129371762275696, 'learning_rate': 0.00016, 'epoch': 0.0}                             
{'loss': 0.8938, 'grad_norm': 0.8135088682174683, 'learning_rate': 0.000152, 'epoch': 0.0}                            
{'loss': 0.7506, 'grad_norm': 0.7057228088378906, 'learning_rate': 0.000144, 'epoch': 0.0}                            
{'loss': 0.8984, 'grad_norm': 0.9443361759185791, 'learning_rate': 0.00013600000000000003, 'epoch': 0.0}              
{'loss': 0.902, 'grad_norm': 0.48894259333610535, 'learning_rate': 0.00012800000000000002, 'epoch': 0.0}              
{'loss': 0.4911, 'grad_norm': 0.5595915913581848, 'learning_rate': 0.00012, 'epoch': 0.0}                             
{'loss': 0.523, 'grad_norm': 0.3571050465106964, 'learning_rate': 0.00011200000000000001, 'epoch': 0.0}               
{'loss': 0.684, 'grad_norm': 0.607795238494873, 'learning_rate': 0.00010400000000000001, 'epoch': 0.0}                
{'loss': 0.6075, 'grad_norm': 0.33529335260391235, 'learning_rate': 9.6e-05, 'epoch': 0.0}                            
{'loss': 0.5018, 'grad_norm': 0.3492335379123688, 'learning_rate': 8.800000000000001e-05, 'epoch': 0.0}               
{'loss': 0.4739, 'grad_norm': 0.5164461135864258, 'learning_rate': 8e-05, 'epoch': 0.0}                               
{'loss': 0.5642, 'grad_norm': 0.3903680443763733, 'learning_rate': 7.2e-05, 'epoch': 0.0}                             
{'loss': 0.5457, 'grad_norm': 0.33454033732414246, 'learning_rate': 6.400000000000001e-05, 'epoch': 0.0}              
{'loss': 0.667, 'grad_norm': 0.37590643763542175, 'learning_rate': 5.6000000000000006e-05, 'epoch': 0.0}              
{'loss': 0.6643, 'grad_norm': 0.3750135600566864, 'learning_rate': 4.8e-05, 'epoch': 0.0}                             
{'loss': 0.7699, 'grad_norm': 0.47293713688850403, 'learning_rate': 4e-05, 'epoch': 0.0}                              
{'loss': 0.5734, 'grad_norm': 0.4119352102279663, 'learning_rate': 3.2000000000000005e-05, 'epoch': 0.0}              
{'loss': 0.5678, 'grad_norm': 0.35942959785461426, 'learning_rate': 2.4e-05, 'epoch': 0.0}                            
{'loss': 0.5165, 'grad_norm': 0.31679877638816833, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.0}             
{'loss': 0.5328, 'grad_norm': 0.3529733717441559, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.0}               
{'train_runtime': 35.2391, 'train_samples_per_second': 2.554, 'train_steps_per_second': 0.851, 'train_loss': 1.2476736972729365, 'epoch': 0.0}
100%|█████████████████████████████████████████████████████████████████████████████████| 30/30 [00:35<00:00,  1.17s/it]
[PID 127, Rank 2] trainer.train() completed.
[PID 126, Rank 1] trainer.train() completed.
[PID 125, Rank 0] trainer.train() completed.
[PID 127, Rank 2] All processes finished training and synchronized.[PID 126, Rank 1] All processes finished training and synchronized.[PID 125, Rank 0] All processes finished training and synchronized.


[PID 127, Rank 2] Script finished successfully for this process.
[PID 126, Rank 1] Script finished successfully for this process.[PID 125, Rank 0] MAIN PROCESS: Training finished. Saving artifacts...

[PID 125, Rank 0] MAIN PROCESS: Saving LoRA adapters to /app/output/lora_adapters_final...
[PID 126] Script __main__ exiting normally.
[PID 127] Script __main__ exiting normally.
[PID 125, Rank 0] MAIN PROCESS: LoRA adapters saved.
[PID 125, Rank 0] MAIN PROCESS: Preparing for full save. Unwrapping model...
[PID 125, Rank 0] MAIN PROCESS: Moving unwrapped model to CPU...
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:126:957 [1] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:126:594 [1] NCCL INFO misc/socket.cc:881 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:127:959 [2] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:127:593 [2] NCCL INFO misc/socket.cc:881 -> 3
7841b93df7f1:126:594 [1] NCCL INFO misc/socket.cc:881 -> 3
7841b93df7f1:126:957 [1] NCCL INFO comm 0x57451020 rank 1 nranks 3 cudaDev 1 busId c2000 - Abort COMPLETE
7841b93df7f1:127:959 [2] NCCL INFO comm 0x370962c0 rank 2 nranks 3 cudaDev 2 busId e1000 - Abort COMPLETE
[PID 125, Rank 0] MAIN PROCESS: Unwrapped model moved to CPU.
[PID 125, Rank 0] MAIN PROCESS: Forcing GC + emptying CUDA cache...
[PID 125, Rank 0] MAIN PROCESS: Merging LoRA into base model (on CPU)...
/opt/venv/lib/python3.12/site-packages/peft/tuners/lora/bnb.py:351: UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors.
  warnings.warn(
[PID 125, Rank 0] MAIN PROCESS: Merge successful.
[PID 125, Rank 0] MAIN PROCESS: Saving merged 16-bit model to /app/output/merged_model_16bit...
Unsloth: Merging 4bit and LoRA weights to 16bit...
Unsloth: Will use up to 159.39 out of 377.26 RAM for saving.
Unsloth: Saving model... This might take 5 minutes ...
  0%|                                                                                          | 0/64 [00:00<?, ?it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  2%|█▎                                                                                | 1/64 [00:00<00:23,  2.68it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  3%|██▌                                                                               | 2/64 [00:00<00:22,  2.72it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  5%|███▊                                                                              | 3/64 [00:01<00:22,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  6%|█████▏                                                                            | 4/64 [00:01<00:21,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  8%|██████▍                                                                           | 5/64 [00:01<00:21,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
  9%|███████▋                                                                          | 6/64 [00:02<00:21,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 11%|████████▉                                                                         | 7/64 [00:02<00:20,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 12%|██████████▎                                                                       | 8/64 [00:02<00:20,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 14%|███████████▌                                                                      | 9/64 [00:03<00:20,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 16%|████████████▋                                                                    | 10/64 [00:03<00:19,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 17%|█████████████▉                                                                   | 11/64 [00:04<00:19,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 19%|███████████████▏                                                                 | 12/64 [00:04<00:18,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 20%|████████████████▍                                                                | 13/64 [00:04<00:18,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 22%|█████████████████▋                                                               | 14/64 [00:05<00:18,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 23%|██████████████████▉                                                              | 15/64 [00:05<00:17,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 25%|████████████████████▎                                                            | 16/64 [00:05<00:17,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 27%|█████████████████████▌                                                           | 17/64 [00:06<00:17,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 28%|██████████████████████▊                                                          | 18/64 [00:06<00:16,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 30%|████████████████████████                                                         | 19/64 [00:06<00:16,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 31%|█████████████████████████▎                                                       | 20/64 [00:07<00:16,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 33%|██████████████████████████▌                                                      | 21/64 [00:07<00:15,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 34%|███████████████████████████▊                                                     | 22/64 [00:08<00:15,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 36%|█████████████████████████████                                                    | 23/64 [00:08<00:14,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 38%|██████████████████████████████▍                                                  | 24/64 [00:08<00:14,  2.76it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 39%|███████████████████████████████▋                                                 | 25/64 [00:09<00:14,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 41%|████████████████████████████████▉                                                | 26/64 [00:09<00:13,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 42%|██████████████████████████████████▏                                              | 27/64 [00:09<00:13,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 44%|███████████████████████████████████▍                                             | 28/64 [00:10<00:13,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 45%|████████████████████████████████████▋                                            | 29/64 [00:10<00:12,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 47%|█████████████████████████████████████▉                                           | 30/64 [00:10<00:12,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 48%|███████████████████████████████████████▏                                         | 31/64 [00:11<00:12,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 50%|████████████████████████████████████████▌                                        | 32/64 [00:11<00:11,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 52%|█████████████████████████████████████████▊                                       | 33/64 [00:12<00:11,  2.75it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 53%|███████████████████████████████████████████                                      | 34/64 [00:12<00:10,  2.74it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 55%|████████████████████████████████████████████▎                                    | 35/64 [00:12<00:10,  2.73it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 56%|█████████████████████████████████████████████▌                                   | 36/64 [00:13<00:10,  2.72it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 58%|██████████████████████████████████████████████▊                                  | 37/64 [00:13<00:09,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 59%|████████████████████████████████████████████████                                 | 38/64 [00:13<00:09,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 61%|█████████████████████████████████████████████████▎                               | 39/64 [00:14<00:09,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 62%|██████████████████████████████████████████████████▋                              | 40/64 [00:14<00:08,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 64%|███████████████████████████████████████████████████▉                             | 41/64 [00:14<00:08,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 66%|█████████████████████████████████████████████████████▏                           | 42/64 [00:15<00:08,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 67%|██████████████████████████████████████████████████████▍                          | 43/64 [00:15<00:07,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 69%|███████████████████████████████████████████████████████▋                         | 44/64 [00:16<00:07,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 70%|████████████████████████████████████████████████████████▉                        | 45/64 [00:16<00:07,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 72%|██████████████████████████████████████████████████████████▏                      | 46/64 [00:16<00:06,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 73%|███████████████████████████████████████████████████████████▍                     | 47/64 [00:17<00:06,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 75%|████████████████████████████████████████████████████████████▊                    | 48/64 [00:17<00:05,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 77%|██████████████████████████████████████████████████████████████                   | 49/64 [00:17<00:05,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 78%|███████████████████████████████████████████████████████████████▎                 | 50/64 [00:18<00:05,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 80%|████████████████████████████████████████████████████████████████▌                | 51/64 [00:18<00:04,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 81%|█████████████████████████████████████████████████████████████████▊               | 52/64 [00:19<00:04,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 83%|███████████████████████████████████████████████████████████████████              | 53/64 [00:19<00:04,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 84%|████████████████████████████████████████████████████████████████████▎            | 54/64 [00:19<00:03,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 86%|█████████████████████████████████████████████████████████████████████▌           | 55/64 [00:20<00:03,  2.70it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 88%|██████████████████████████████████████████████████████████████████████▉          | 56/64 [00:20<00:02,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 89%|████████████████████████████████████████████████████████████████████████▏        | 57/64 [00:20<00:02,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 91%|█████████████████████████████████████████████████████████████████████████▍       | 58/64 [00:21<00:02,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 92%|██████████████████████████████████████████████████████████████████████████▋      | 59/64 [00:21<00:01,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 94%|███████████████████████████████████████████████████████████████████████████▉     | 60/64 [00:22<00:01,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 95%|█████████████████████████████████████████████████████████████████████████████▏   | 61/64 [00:22<00:01,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 97%|██████████████████████████████████████████████████████████████████████████████▍  | 62/64 [00:22<00:00,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
 98%|███████████████████████████████████████████████████████████████████████████████▋ | 63/64 [00:23<00:00,  2.71it/s][WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
[WARN] fast_dequantize() skipping: model is on CPU. Verifying shape before reshaping.
100%|█████████████████████████████████████████████████████████████████████████████████| 64/64 [00:23<00:00,  2.72it/s]
Unsloth: Saving tokenizer... Done.

Done.
[PID 125, Rank 0] MAIN PROCESS: Merged 16-bit model saved.
[PID 125, Rank 0] MAIN PROCESS: Saving model to GGUF (from CPU copy) to /app/output/full_merged_model...
[PID 125, Rank 0] MAIN PROCESS: Model saved to GGUF via save_pretrained.
[PID 125, Rank 0] MAIN PROCESS: Deleting trainer and model references to free VRAM...
[PID 125, Rank 0] MAIN PROCESS: Forcing garbage collection and emptying CUDA cache... (Post-Save)
[PID 125, Rank 0] Script finished successfully for this process.
[PID 125, MainRank] __main__: Training complete. Metrics: TrainOutput(global_step=30, training_loss=1.2476736972729365, metrics={'train_runtime': 35.2391, 'train_samples_per_second': 2.554, 'train_steps_per_second': 0.851, 'total_flos': 4126896912596992.0, 'train_loss': 1.2476736972729365})
[PID 125] Script __main__ exiting normally.
[rank0]:[W521 02:27:23.998977279 ProcessGroupNCCL.cpp:1535] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:125:595 [0] NCCL INFO misc/socket.cc:881 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:64 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:80 -> 3
7841b93df7f1:125:962 [0] NCCL INFO misc/socket.cc:829 -> 3
7841b93df7f1:125:962 [0] NCCL INFO comm 0x4e244b50 rank 0 nranks 3 cudaDev 0 busId c1000 - Abort COMPLETE