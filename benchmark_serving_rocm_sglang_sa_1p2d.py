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
    print(f'config, tp, conc, prompts, mnbt, isl, osl, rqr, req, e2el, ttft, tpot, itl, p90ttft, p90tpot, p90itl, output_tput, total_tput')


def launch_bmk_llama(model_name, input_len, output_len, tp_size, max_concurrency, max_num_batched_tokens, request_rate):

    # 1p1d 
    psize = 1
    dsize = 2
    ori_tp_size = tp_size
    tp_size = int(tp_size / (psize + dsize))


    resultpath = f"/billhe/{psize}p{dsize}d-results"
    outsidepath = f"/home/amd/billhe/{psize}p{dsize}d-results"

    if model_name == '/models/amd_Llama-3.3-70B-Instruct-FP8-KV':
        model_code = '70b'
    elif model_name == '/models/amd_Llama-3.3-405B-Instruct-FP8-KV':
        model_code = '405b'
    else:
        raise ValueError(f'{model_name} not supported')

    result_filename = (
            f'{model_code}_tp{ori_tp_size}_{psize}p{dsize}d_isl{input_len}_osl{output_len}_'
            f'c{max_concurrency}_mnbt{max_num_batched_tokens}_rps{request_rate}'
    )
    result_file_path = Path(f'{outsidepath}/{result_filename}.json')

    if args.print_results:
        if not result_file_path.exists():
            return
        fields = ['request_throughput', 'median_e2el_ms', 'median_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'p90_ttft_ms',\
                  'p90_tpot_ms', 'p90_itl_ms', 'output_throughput', 'total_token_throughput']
        with open(result_file_path) as f:
            results = json.load(f)
        # print(f'{result_filename}, {tp_size}, {max_concurrency}, {max_concurrency*2}, {max_num_batched_tokens}, {input_len}, {output_len}, \
            #   {request_rate}, ', ', '.join(f'{results[f]:.3f}' for f in fields))
        print(
            f'{result_filename:>50}, '
            f'{ori_tp_size:>4}, '
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

    network_name = 'bmk-net'
    serverp_name = 'bmk-server-p'
    serverd_name = 'bmk-server-d'
    serverlb_name = 'bmk-server-lb'
    port = 30501
    image_name = 'rocm/ali-private:sglang-v0.4.7-rocm630-deepep-bcm-0625'
    client_image_name = 'rocm/vllm-dev:nightly_main_20250706'
    range_ratio = 0.9

    metric_percentiles = ','.join([str(i) for i in range(1, 100, 1)])

    script = f'''#!/usr/bin/env bash
docker network rm -f {network_name}
docker network create {network_name}

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
        --host 0.0.0.0             \
        --port {port}              \
        --mem-fraction-static 0.9               \
        --disable-radix-cache       \
        --tp-size {tp_size}         \
        --base-gpu-id 0      \
        --max-running-requests 1024 \
        --disaggregation-mode prefill \
	    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

printf "RESULT_FILENAME=%s\n" "{result_filename}"
while ! docker logs {serverp_name} 2>&1 | grep -q "The server is fired up and ready to roll"; do
    sleep 1
    if docker logs {serverp_name} 2>&1 | grep -q "ERROR"; then
        docker logs {serverp_name} >& "failed_runs/{result_filename}.log"
        docker stop {serverp_name};docker network rm {network_name}
        exit 1
    fi
done


docker run --rm -d --network {network_name} --ipc host --name {serverd_name}1 \
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
        --host 0.0.0.0             \
        --port {port}              \
        --mem-fraction-static 0.9               \
        --disable-radix-cache       \
        --tp-size {tp_size}         \
        --base-gpu-id 2      \
        --max-running-requests 1024 \
        --disaggregation-mode decode \
	    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

printf "RESULT_FILENAME=%s\n" "{result_filename}"
while ! docker logs {serverd_name}1 2>&1 | grep -q "The server is fired up and ready to roll"; do
    sleep 1
    if docker logs {serverd_name}1 2>&1 | grep -q "ERROR"; then
        docker logs {serverd_name}1 >& "failed_runs/{result_filename}.log"
        docker stop {serverd_name}1;docker network rm {network_name}
        exit 1
    fi
done

docker run --rm -d --network {network_name} --ipc host --name {serverd_name}2 \
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
        --host 0.0.0.0             \
        --port {port}              \
        --mem-fraction-static 0.9               \
        --disable-radix-cache       \
        --tp-size {tp_size}         \
        --base-gpu-id 4      \
        --max-running-requests 1024 \
        --disaggregation-mode decode \
	    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

printf "RESULT_FILENAME=%s\n" "{result_filename}"
while ! docker logs {serverd_name}2 2>&1 | grep -q "The server is fired up and ready to roll"; do
    sleep 1
    if docker logs {serverd_name}2 2>&1 | grep -q "ERROR"; then
        docker logs {serverd_name}2 >& "failed_runs/{result_filename}.log"
        docker stop {serverd_name}2;docker network rm {network_name}
        exit 1
    fi
done


docker run --rm -d --network {network_name} --ipc host --name {serverlb_name} \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe \
    {image_name} \
    python -m sglang.srt.disaggregation.mini_lb --prefill http://{serverp_name}:{port} --decode http://{serverd_name}1:{port} http://{serverd_name}2:{port} --host 0.0.0.0 --port {port}


docker run --rm -t --network {network_name} --name bmk-client \
    --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HUGGINGFACE_HUB_CACHE=/models -e MODELSCOPE_CACHE=/models \
    -v /home/amd/models:/models -v /home/amd/billhe:/billhe --workdir /billhe/vllm-upstream/benchmarks \
    {client_image_name} \
        python benchmark_serving.py \
            --backend sglang \
            --base-url "http://{serverlb_name}:{port}" \
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
            --save-result --result-dir "{resultpath}" --result-filename "{result_filename}.json"

docker stop {serverp_name};docker stop {serverd_name}1;docker stop {serverd_name}2;docker stop {serverlb_name};docker network rm {network_name}
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
        for tp_size in [8, ]: 
            for max_concurrency in [128, ]:
                for qps in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
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
