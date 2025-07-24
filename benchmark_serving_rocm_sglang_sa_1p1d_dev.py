# benchmark script from https://gist.githubusercontent.com/kimbochen/15aab40c7a00613f8aa400d157d0ffd1/raw/06c2d2b67337836ec1919ccc6ffd75dae541dfed/amd_bmk.py

import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, choices=['mi300x', 'mi308x'], default="mi300x")
parser.add_argument('-p', '--print-results', action='store_true', default=False)
args = parser.parse_args()

if args.print_results:
    # print(f'config, tp, conc, prompts, mnbt, isl, osl, rqr, req, goodput, e2el, ttft, tpot, itl, p90ttft, p90tpot, p90itl, output_tput, total_tput')
    print(f'config, tp, conc, prompts, mnbt, isl, osl, rqr, req, goodput, p95ttft, p95tpot, p95itl, p99ttft, p99tpot, p99itl, output_tput, total_tput')


def launch_bmk_llama(model_name, input_len, output_len, tp_size, max_concurrency, max_num_batched_tokens, request_rate):

    # 1p2d 
    psize = 1
    dsize = 1

    enable_goodput = "_goodput"
    goodput_metric = "tpot:25"


    resultpath = f"/billhe/{psize}p{dsize}d-results-dev"
    outsidepath = f"/home/amd/billhe/{psize}p{dsize}d-results"

    if model_name == '/models/amd_Llama-3.3-70B-Instruct-FP8-KV':
        model_code = '70b'
    elif model_name == '/models/amd_Llama-3.1-405B-Instruct-FP8-KV':
        model_code = '405b'
    else:
        raise ValueError(f'{model_name} not supported')

    result_filename = (
            f'{model_code}_tp{tp_size}_{psize}p{dsize}d_isl{input_len}_osl{output_len}_'
            f'c{max_concurrency}_mnbt{max_num_batched_tokens}_rps{request_rate}{enable_goodput}_{goodput_metric}'
    )
    result_file_path = Path(f'{outsidepath}/{result_filename}.json')

    if args.print_results:
        if not result_file_path.exists():
            return
        # fields = ['request_throughput', 'request_goodput', 'median_e2el_ms', 'median_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'p90_ttft_ms',\
                #   'p90_tpot_ms', 'p90_itl_ms', 'output_throughput', 'total_token_throughput']
        fields = ['request_throughput', 'request_goodput', 'p95_ttft_ms','p95_tpot_ms', 'p95_itl_ms', 'p99_ttft_ms','p99_tpot_ms', 'p99_itl_ms', 'output_throughput', 'total_token_throughput']
        
        with open(result_file_path) as f:
            results = json.load(f)
        # print(f'{result_filename}, {tp_size}, {max_concurrency}, {max_concurrency*2}, {max_num_batched_tokens}, {input_len}, {output_len}, \
            #   {request_rate}, ', ', '.join(f'{results[f]:.3f}' for f in fields))
        print(
            f'{result_filename:>50}, '
            f'{tp_size:>4}, '
            f'{max_concurrency:>4}, '
            f'{max_concurrency * 2:>4}, '
            f'{max_num_batched_tokens:>6}, '
            f'{input_len:>4}, '
            f'{output_len:>4}, '
            f'{request_rate:>6.1f}, ' +
            ', '.join(f'{results[f]:>12.3f}' for f in fields)
        )
        return

    if result_file_path.exists():
        print(f'Skipping {result_filename}')
        return

    # network_name = 'bmk-net'
    network_name = 'host'
    serverp_name = 'bmk-server-p'
    serverp_ip = 'tw015'
    pport = 30501

    serverd1_name = 'bmk-server-d1'
    serverd1_ip = 'tw033'
    d1port = 30502
    
    serverlb_name = 'bmk-server-lb'
    serverlb_ip = 'tw015'
    lbport = 30500
    
    image_name = 'rocm/ali-private:sglang-v0.4.7-rocm630-deepep-bcm-0625'
    client_image_name = 'rocm/vllm-dev:nightly_main_20250706'
    range_ratio = 0.9

    metric_percentiles = ','.join([str(i) for i in range(1, 100, 1)])

    script = f'''#!/bin/bash

ssh -i /home/amd/.ssh/id_rsa {serverp_ip}                                    \
docker run --rm -d --network {network_name} --ipc host --name {serverp_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe \
    {image_name} \
    python3 -m sglang.launch_server \
        --model {model_name}     \
        --trust-remote-code             \
        --chunked-prefill-size -1  \
        --max-prefill-tokens 2048 \
        --stream-output \
        --host {serverp_ip}             \
        --port {pport}              \
        --mem-fraction-static 0.9               \
        --disable-radix-cache       \
        --tp-size {tp_size}         \
        --base-gpu-id 4      \
        --max-running-requests 1024 \
        --disaggregation-mode prefill \
	    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

printf "RESULT_FILENAME=%s\n" "{result_filename}"

timeout_seconds=600
start_time=$(date +%s)
while true; do
    if curl -s "{serverp_ip}:{pport}/v1/models" > /dev/null; then
      echo "Server {serverp_ip} on port {pport} is ready."
      break
    fi
    now=$(date +%s)
    if [ $((now - start_time)) -ge $timeout_seconds ]; then
      echo "Timeout waiting for server {serverp_ip} on port {pport}"
      exit 1
    fi
    sleep 1
done


ssh -i /home/amd/.ssh/id_rsa {serverd1_ip}                                    \
docker run --rm -d --network {network_name} --ipc host --name {serverd1_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe \
    {image_name} \
    python3 -m sglang.launch_server \
        --model {model_name}     \
        --trust-remote-code             \
        --chunked-prefill-size -1  \
        --max-prefill-tokens 2048 \
        --stream-output \
        --host {serverd1_ip}             \
        --port {d1port}              \
        --mem-fraction-static 0.9               \
        --disable-radix-cache       \
        --tp-size {tp_size}         \
        --base-gpu-id 0      \
        --max-running-requests 1024 \
        --disaggregation-mode decode \
	    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

timeout_seconds=600
start_time=$(date +%s)
while true; do
    if curl -s "{serverd1_ip}:{d1port}/v1/models" > /dev/null; then
      echo "Server {serverd1_ip} on port {d1port} is ready."
      break
    fi
    now=$(date +%s)
    if [ $((now - start_time)) -ge $timeout_seconds ]; then
      echo "Timeout waiting for server {serverd1_ip} on port {d1port}"
      exit 1
    fi
    sleep 1
done


ssh -i /home/amd/.ssh/id_rsa {serverlb_ip}                                    \
docker run --rm -d --network {network_name} --ipc host --name {serverlb_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe \
    {image_name} \
    python -m sglang.srt.disaggregation.mini_lb --prefill http://{serverp_ip}:{pport} --decode http://{serverd1_ip}:{d1port} --host {serverlb_ip} --port {lbport}

sleep 5
    
ssh -i /home/amd/.ssh/id_rsa {serverlb_ip}                                    \
docker run --rm -t --network {network_name} --name bmk-client \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe/vllm-upstream/benchmarks \
    {client_image_name} \
        python benchmark_serving.py \
            --backend sglang \
            --base-url "http://{serverlb_ip}:{lbport}" \
            --model {model_name} \
            --percentile-metrics "ttft,tpot,itl,e2el" \
			--metric-percentiles {metric_percentiles} \
            --request-rate {request_rate} \
            --ignore-eos \
            --max-concurrency {max_concurrency} \
            --dataset-name random \
            --random-input-len {input_len} \
            --random-output-len {output_len} \
            --random-range-ratio {range_ratio} \
            --num-prompts $(( {max_concurrency} * 2 )) \
            --goodput {goodput_metric} \
            --save-result --result-dir "{resultpath}" --result-filename "{result_filename}.json"

ssh -i /home/amd/.ssh/id_rsa {serverp_ip} docker stop {serverp_name}
ssh -i /home/amd/.ssh/id_rsa {serverd1_ip} docker stop {serverd1_name}
ssh -i /home/amd/.ssh/id_rsa {serverlb_ip} docker stop {serverlb_name}

sleep 5
'''
    
    print(f"Executing script: {script}")
    subprocess.run(script, shell=True, check=True)


# def launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency):
#     result_filename = f'dsv3_tp{tp_size}_isl{input_len}_osl{output_len}_c{max_concurrency}'
#     result_file_path = Path(f'results/{result_filename}.json')

#     if args.print_results:
#         if not result_file_path.exists():
#             return
#         fields = ['median_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'median_e2el_ms', 'total_token_throughput']
#         with open(result_file_path) as f:
#             results = json.load(f)
#         print(f'{result_filename}, {tp_size}, {max_concurrency}, -1,', ', '.join(f'{results[f]:.3f}' for f in fields))
#         return

#     if result_file_path.exists():
#         print(f'Skipping {result_filename}')
#         return

#     model_name = 'deepseek-ai/DeepSeek-V3'
#     network_name = 'bmk-net'
#     server_name = 'bmk-server'
#     port = 8000
#     image_name = 'rocm/sgl-dev:upstream_20250422'

#     script = f'''#!/usr/bin/env bash
# docker network create {network_name}

# docker run --rm -d --network {network_name} --ipc host --name {server_name} \
#     --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
#     --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v "$PWD/.hf_cache/":/root/hf_cache/ -v "$PWD/.inductor_cache/":/tmp/torchinductor_root/ \
#     -e HF_HUB_CACHE=/root/hf_cache/ -e HF_TOKEN="$(cat hf_token.txt)" -e SGLANG_AITER_MOE=1 \
#     {image_name} \
#     python3 -m sglang.launch_server --model-path {model_name} --host 0.0.0.0 --port {port} --tp {tp_size} --trust-remote-code \
#     --chunked-prefill-size 131072 --enable-torch-compile --torch-compile-max-bs 256

# printf "RESULT_FILENAME=%s\n" "{result_filename}"
# while ! docker logs {server_name} 2>&1 | grep -q "The server is fired up and ready to roll!"; do
#     sleep 1
# done

# docker run --rm -t --network {network_name} --name bmk-client \
#     --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
#     --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v $PWD:/workspace/ -w /workspace/vllm/benchmarks/ -e HF_TOKEN=$(cat hf_token.txt) \
#     rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250410 \
#         python benchmark_serving.py \
#             --model {model_name} --backend vllm --base-url "http://{server_name}:{port}" \
#             --dataset-name "random" --random-input-len {input_len} --random-output-len {output_len} --random-prefix-len 0 \
#             --num-prompts $(( {max_concurrency} * 10 )) --max-concurrency {max_concurrency} --request-rate "inf" --ignore-eos \
#             --save-result --result-dir "/workspace/results/" --result-filename "{result_filename}.json" --percentile-metrics "ttft,tpot,itl,e2el"

# docker stop {server_name}; docker network rm {network_name}
# sleep 60
# '''
#     subprocess.run(script, shell=True, check=True)


if args.gpu == 'mi300x':
    max_num_batched_tokens = 65536

    # for input_len, output_len in [(1024, 1024), (1024, 4096), (4096, 1024)]:
    for input_len, output_len in [(3200, 800), (1024, 2048), ]:
        t_s = time.time()
        # LLaMA 70B
        for tp_size in [4, ]: 
            for max_concurrency in [128, ]:
                # for qps in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
                # for qps in list(reversed([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])):
                # for qps in list(reversed([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])):
                for qps in list(reversed([1.8, 2.0])):
                # for qps in [2.0,]:
                    launch_bmk_llama('/models/amd_Llama-3.3-70B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_num_batched_tokens, qps)
        # # LLaMA 405B FP8
        # for tp_size in [4, 8]:
        #     for max_concurrency in [4, 8, 16, 32, 64, 128, 256]: 
        #         launch_bmk_llama('amd/Llama-3.1-405B-Instruct-FP8-KV', input_len, output_len, tp_size, max_concurrency, max_num_batched_tokens)

        # # DeepseekV3
        # tp_size = 8
        # for max_concurrency in [4, 8, 16, 32, 64, 128, 256]: 
        #     launch_bmk_deepseek(input_len, output_len, tp_size, max_concurrency)
        t_e = time.time()
        if not args.print_results:
            print(f'ISL{input_len}/OSL{output_len} BENCHMARK TIME ELAPSED: {((t_e - t_s) / 60.0):.2f} minutes')
else:
    raise ValueError(f'Unknown GPU {args.gpu}')
