eval "$('./miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
export http_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"
export https_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"

export http_proxy="http://127.0.0.1:10808"
export https_proxy="https://127.0.0.1:10808"

export GIT_SSH_COMMAND="ssh -i /mnt/jfzn/yhc/.cache/k"
export GIT_AUTHOR_NAME="chenxw"
export GIT_AUTHOR_EMAIL="severuschen2000@gmail.com"
export GIT_COMMITTER_NAME="chenxw"
export GIT_COMMITTER_EMAIL="severuschen2000@gmail.com"

pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export http_proxy="http://localhost:20171"
export https_proxy="http://localhost:20171"

MODEL_PATH=/mnt/jfzn/yhc/models/meta-llama/Llama-3.1-8B
LD_PRELOAD=/mnt/jfzn/miniconda3/envs/yhc_olmes/lib/libstdc++.so olmes \
    --model model=$MODEL_PATH,tokenizer=$MODEL_PATH,trust_remote_code=True,add_bos_token=True,enforce_eager=True \
    --model-type vllm \
    --task \
        gsm8k::ultra_default \
    --gpus 1 \
    --batch-size 128 \
    --output-dir /mnt/jfzn/yhc/ULTra_Eval/test_vllm/4



cd /mnt/jfzn/yhc/v2ray && V2RAYA_CONFIG=/mnt/jfzn/yhc/v2ray/v2raya v2raya
npm install -g localtunnel
lt --port 2017
curl https://loca.lt/mytunnelpassword

apt install proxychains
proxychains -c