import os
import time
import subprocess
import psutil
import argparse
import cv2
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def extract_frame(video_path, frame_num):
    """Extract a specific frame from video and save as JPG"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num >= total_frames:
        raise ValueError(f"Frame number {frame_num} exceeds video length ({total_frames} frames)")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Failed to extract frame {frame_num}")
        
    frame_path = f"frame_{frame_num}.jpg"
    cv2.imwrite(frame_path, frame)
    return frame_path

def cleanup_frames(num_frames):
    """Clean up extracted frames"""
    for i in range(num_frames):
        frame_path = f"frame_{i}.jpg"
        try:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        except Exception as e:
            logger.warning(f"Failed to remove frame {frame_path}: {str(e)}")

def run_command(command, shell=False, timeout=300):
    """Run command and measure time and memory usage with timeout"""
    logger.info(f"Running command: {command}")
    
    start_mem = get_memory_usage()
    start_time = time.time()
    
    try:
        if shell:
            process = subprocess.Popen(
                command, 
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        else:
            process = subprocess.Popen(
                command.split(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
        
        stdout, stderr = process.communicate(timeout=timeout)
        
        end_time = time.time()
        end_mem = get_memory_usage()
        
        if stdout:
            logger.info(stdout)
        if stderr:
            logger.warning(f"Errors: {stderr}")
            
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
            
        return end_time - start_time, end_mem - start_mem
        
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"Command timed out after {timeout} seconds: {command}")
    except Exception as e:
        raise RuntimeError(f"Command failed: {str(e)}")

def run_inference_on_frames(command_template, num_frames, warmup=False):
    """Run inference on specified number of frames with optional warmup"""
    total_time = 0
    total_mem = 0
    successful_frames = 0
    
    # Perform warmup if requested
    if warmup:
        logger.info("Performing warmup run...")
        try:
            warmup_command = command_template.format("frame_0.jpg")
            run_command(warmup_command)
        except Exception as e:
            logger.warning(f"Warmup failed: {str(e)}")

    # Run actual inference
    for i in range(num_frames):
        frame_path = f"frame_{i}.jpg"
        command = command_template.format(frame_path)
        try:
            time_taken, mem_usage = run_command(command)
            total_time += time_taken
            total_mem += mem_usage
            successful_frames += 1
            logger.info(f"Processed frame {i+1}/{num_frames} - Time: {time_taken:.4f}s")
        except Exception as e:
            logger.error(f"Failed on frame {i}: {str(e)}")

    if successful_frames > 0:
        avg_time = total_time / successful_frames
        avg_mem = total_mem / successful_frames
        return avg_time, avg_mem, successful_frames
    return None, None, 0

def main():
    parser = argparse.ArgumentParser(description='Benchmark FastSAM models')
    parser.add_argument('--video', type=str, required=True,
                      help='path to input video')
    parser.add_argument('--num_frames', type=int, default=10,
                      help='number of frames to process')
    parser.add_argument('--warmup', action='store_true',
                      help='perform warmup run before benchmarking')
    parser.add_argument('--timeout', type=int, default=300,
                      help='timeout in seconds for each command')
    parser.add_argument('--trt-timeout', type=int, default=1800,
                      help='timeout in seconds for TensorRT conversion (default: 30 minutes)')
    args = parser.parse_args()

    # Get absolute paths
    current_dir = os.getcwd()
    video_path = os.path.abspath(args.video)
    pytorch_model = os.path.abspath('FastSAM.pt')
    onnx_model = os.path.abspath('fast_sam_1024.onnx')
    trt_model = os.path.abspath('fast_sam_1024.trt')

    # Validate input files
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    if not os.path.exists(pytorch_model):
        logger.error(f"PyTorch model not found: {pytorch_model}")
        return

    results = []

    try:
        # Extract frames from video
        logger.info(f"Extracting {args.num_frames} frames from video...")
        for i in range(args.num_frames):
            try:
                extract_frame(video_path, i)
            except Exception as e:
                logger.error(f"Failed to extract frame {i}: {str(e)}")
                cleanup_frames(args.num_frames)
                return

        # 1. PyTorch Inference
        logger.info("\n=== Step 1: Running PyTorch Inference ===")
        try:
            pytorch_command = f"python3 FastSAM/Inference.py --model_path {pytorch_model} --img_path {{}}"
            time_taken, mem_usage, successful = run_inference_on_frames(
                pytorch_command, 
                args.num_frames,
                warmup=args.warmup
            )
            if time_taken is not None:
                results.append(("PyTorch", time_taken, mem_usage, successful))
        except Exception as e:
            logger.error(f"PyTorch inference failed: {str(e)}")

        # 2. Convert PyTorch to ONNX
        if not os.path.exists(onnx_model):
            logger.info("\n=== Step 2: Converting PyTorch to ONNX ===")
            try:
                command = f"python3 FastSam_Awsome_TensorRT/pt2onnx.py --weights {pytorch_model} --output {onnx_model}"
                run_command(command, timeout=args.timeout)
            except Exception as e:
                logger.error(f"ONNX conversion failed: {str(e)}")
                cleanup_frames(args.num_frames)
                return
        else:
            logger.info(f"ONNX model already exists: {onnx_model}. Skipping conversion.")

        # 3. ONNX Inference
        if os.path.exists(onnx_model):
            logger.info("\n=== Step 3: Running ONNX Inference ===")
            try:
                onnx_command = f"python3 FastSam_Awsome_TensorRT/infer_onnx.py --model {onnx_model} --img {{}}"
                time_taken, mem_usage, successful = run_inference_on_frames(
                    onnx_command, 
                    args.num_frames,
                    warmup=args.warmup
                )
                if time_taken is not None:
                    results.append(("ONNX", time_taken, mem_usage, successful))
            except Exception as e:
                logger.error(f"ONNX inference failed: {str(e)}")

        # 4. Convert ONNX to TensorRT
        if not os.path.exists(trt_model):
            logger.info("\n=== Step 4: Converting ONNX to TensorRT ===")
            try:
                trt_script = f"""#!/bin/bash
        trtexec --onnx={onnx_model} \
                --saveEngine={trt_model} \
                --fp16 \
                --optShapes=images:1x3x1024x1024 \
                --verbose
        """
                script_path = 'temp_onnx2trt.sh'
                with open(script_path, 'w') as f:
                    f.write(trt_script)
                os.chmod(script_path, 0o755)
                
                # Execute the script
                run_command(f"bash {script_path}", shell=True, timeout=args.trt_timeout)
                os.remove(script_path)
            except Exception as e:
                logger.error(f"TensorRT conversion failed: {str(e)}")
                cleanup_frames(args.num_frames)
                return
        else:
            logger.info(f"TensorRT model already exists: {trt_model}. Skipping conversion.")


        # 5. TensorRT Inference
        if os.path.exists(trt_model):
            logger.info("\n=== Step 5: Running TensorRT Inference ===")
            try:
                trt_command = f"python3 FastSam_Awsome_TensorRT/inference_trt.py --engine {trt_model} --img {{}}"
                time_taken, mem_usage, successful = run_inference_on_frames(
                    trt_command, 
                    args.num_frames,
                    warmup=args.warmup
                )
                if time_taken is not None:
                    results.append(("TensorRT", time_taken, mem_usage, successful))
            except Exception as e:
                logger.error(f"TensorRT inference failed: {str(e)}")

    finally:
        # Clean up frames
        cleanup_frames(args.num_frames)

    # Print results
    if results:
        logger.info("\nPerformance Comparison:")
        logger.info("Model     | Avg Time/Frame | Memory Usage (MB) | FPS    | Success")
        logger.info("-" * 65)
        for model, time_taken, mem_usage, successful in results:
            fps = 1.0 / time_taken if time_taken > 0 else 0
            success_rate = (successful / args.num_frames) * 100
            logger.info(f"{model:<9} | {time_taken:11.4f}s | {mem_usage:14.2f} | {fps:6.2f} | {success_rate:6.1f}%")
    else:
        logger.info("\nNo successful benchmarks to compare.")

if __name__ == "__main__":
    main()
