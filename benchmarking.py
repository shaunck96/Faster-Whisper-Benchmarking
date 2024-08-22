import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pandas as pd
import time
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import ray

# Global model variable
global_transcription_model = None

# Function to initialize the model globally
def init_global_model(device_index):
    global global_transcription_model
    if global_transcription_model is None:
        global_transcription_model = WhisperModel("medium.en", 
                                                  device="cuda", 
                                                  device_index=device_index, 
                                                  compute_type="float16")
    return global_transcription_model

# Helper function for benchmarking
def print_benchmark_results(start_time, end_time, model_type):
    total_time = end_time - start_time
    print(f"{model_type} Inference completed in {total_time:.2f} seconds.")
    print("GPU Memory Usage:")
    os.system("nvidia-smi")

# Multi-GPU Parallelism (Current Approach)
class Transcribe:
    def __init__(self, device_index):
        self.device_index = device_index
        self.transcription_model = init_global_model(self.device_index)  # Use the global model

    def predict(self, model_input):
        try:
            segments, _ = self.transcription_model.transcribe(
                model_input,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            return segments
        except Exception as e:
            print(f"Error in transcription on GPU {self.device_index}: {str(e)}")
            return []

def transcribe_files(audio_files, device_index):
    model = Transcribe(device_index)
    transcriptions = []
    for audio_file in audio_files:
        print(f"Transcribing file {audio_file} on GPU {device_index}...")
        start_time = time.time()
        segments = model.predict(audio_file)
        end_time = time.time()
        transcription_time = end_time - start_time

        file_transcription = [{
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'no_speech_probability': segment.no_speech_prob
        } for segment in segments]

        transcriptions.append({
            'file': audio_file,
            'transcription': file_transcription,
            'transcription_time': transcription_time
        })
    return transcriptions

def multi_gpu_parallelism(file_names):
    half_size = len(file_names) // 2
    files_gpu_0 = file_names[:half_size]
    files_gpu_1 = file_names[half_size:]

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transcribe_files, files_gpu_0, 0),
            executor.submit(transcribe_files, files_gpu_1, 1)
        ]
        results = [future.result() for future in futures]

    return results[0] + results[1]

# PyTorch DataParallel
class TranscribeDataParallel:
    def __init__(self):
        self.transcription_model = None
        self.load_context()

    def load_context(self):
        self.transcription_model = WhisperModel("medium.en", 
                                                device="cuda", 
                                                device_index=[0, 1],  # Use a list for multi-GPU
                                                compute_type="float16")
        self.transcription_model = torch.nn.DataParallel(self.transcription_model)

    def predict(self, model_input):
        try:
            segments, _ = self.transcription_model.module.transcribe(
                model_input,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            return segments
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return []

def transcribe_files_dataparallel(audio_files):
    model = TranscribeDataParallel()
    transcriptions = []
    for audio_file in audio_files:
        print(f"Transcribing file {audio_file} using DataParallel...")
        start_time = time.time()
        segments = model.predict(audio_file)
        end_time = time.time()
        transcription_time = end_time - start_time

        file_transcription = [{
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'no_speech_probability': segment.no_speech_prob
        } for segment in segments]

        transcriptions.append({
            'file': audio_file,
            'transcription': file_transcription,
            'transcription_time': transcription_time
        })
    return transcriptions

# Distributed Data Parallel (DDP)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def transcribe_ddp(rank, world_size, file_names):
    setup(rank, world_size)
    model = Transcribe(rank)
    transcriptions = transcribe_files(file_names, rank)
    dist.destroy_process_group()
    return transcriptions

def distributed_data_parallel(file_names):
    world_size = 2
    half_size = len(file_names) // 2
    files_gpu_0 = file_names[:half_size]
    files_gpu_1 = file_names[half_size:]

    mp.spawn(transcribe_ddp, args=(world_size, files_gpu_0), nprocs=world_size, join=True)

# Ray Parallelism
@ray.remote(num_gpus=1)
def transcribe_files_ray(audio_files, device_index):
    return transcribe_files(audio_files, device_index)

def ray_parallelism(file_names):
    ray.init(ignore_reinit_error=True)

    half_size = len(file_names) // 2
    files_gpu_0 = file_names[:half_size]
    files_gpu_1 = file_names[half_size:]

    futures = [transcribe_files_ray.remote(files_gpu_0, 0), transcribe_files_ray.remote(files_gpu_1, 1)]
    results = ray.get(futures)

    ray.shutdown()

    return results[0] + results[1]

def main():
    today = datetime.now().date() - timedelta(days=2)
    dbfs_path = "audio_files/"
    file_names = [os.path.join(dbfs_path, f) for f in os.listdir(dbfs_path)][:20]

    # 1. Multi-GPU Parallelism (ProcessPoolExecutor)
    start_time = time.time()
    multi_gpu_result = multi_gpu_parallelism(file_names)
    end_time = time.time()
    print_benchmark_results(start_time, end_time, "Multi-GPU Parallelism")

    # 2. DataParallel
    start_time = time.time()
    dataparallel_result = transcribe_files_dataparallel(file_names)
    end_time = time.time()
    print_benchmark_results(start_time, end_time, "DataParallel")

    # 3. Distributed Data Parallel (DDP)
    start_time = time.time()
    distributed_data_parallel(file_names)
    end_time = time.time()
    print_benchmark_results(start_time, end_time, "Distributed Data Parallel")

    # 4. Ray Parallelism
    start_time = time.time()
    ray_result = ray_parallelism(file_names)
    end_time = time.time()
    print_benchmark_results(start_time, end_time, "Ray Parallelism")

    # Combine results
    all_transcriptions = multi_gpu_result + dataparallel_result + ray_result
    transcription_df = pd.DataFrame(all_transcriptions)

    # Save the DataFrame to Parquet format
    save_path = f'gpu_transcriptions_benchmarking.parquet'
    transcription_df.to_parquet(save_path, index=False)
    print(f"DataFrame successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
