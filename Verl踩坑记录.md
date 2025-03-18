## 数据准备

数据处理的代码路径

```
verl/examples/data_preprocess
```

官方文档数据处理方式

```bash
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

在服务器上会遇到 HF 网站访问被墙的问题，可以采用如下方式改成从镜像网站下载：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

下载的时候可以根据自己需要修改文件保存的路径，例如：
```python
parser.add_argument('--local_dir', type=str, default='/verl/data/full_hh_rlhf')
```

或者直接采用以下下载脚本：
```bash
export HF_ENDPOINT=https://hf-mirror.com
base_path=Your Path Here

python3 examples/data_preprocess/gsm8k.py --local_dir ${base_path}/data/gsm8k
```

## 下载模型

模型下载的两种方式

#### 方案1: 镜像 HF

##### 镜像hf下载脚本

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model --resume-download XXXX  --local-dir ${base_path}/XXXX
```

##### 镜像hf代码下载

```python
import os
import argparse
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser = argparse.ArgumentParser(description="Download a transformers model from huggingface hub.")
parser.add_argument("--model_name", "-m", type=str, default="Huggingface model name", required=True)
parser.add_argument("--dir", type=str, default=None, help="Local dir to save the model")
parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache dir")
args = parser.parse_args()

if args.dir is not None:
    args.dir = os.path.join(args.dir, args.model_name)
os.makedirs(args.dir, exist_ok=True)

snapshot_download(repo_id=args.model_name, 
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors", "*.onnx"],
                  local_dir=args.dir, 
                  local_dir_use_symlinks=False,
                  resume_download=True,
                  cache_dir=args.cache_dir)
```

下载示例

```bash
export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1

model_name=morecry/BaichuanCharRM
dir=/.cache/huggingface/pretrained_model
cache_dir=/.cache/huggingface/pretrained_model/

python download_model.py -m ${model_name} --dir ${dir} --cache_dir ${cache_dir}
```

#### 方案2: ModelScope 下载

```python
from modelscope import snapshot_download
   model_ids=[]
   for model_id in model_ids:
       model_dir = snapshot_download(model_id)
```

创建 `modelscope_downloader.py` 文件

```python
from modelscope import snapshot_download
import argparse
import os

parser = argparse.ArgumentParser(description="Download a transformers model from modelscope.")
parser.add_argument("--model_id", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", required=True)
parser.add_argument("--dir", type=str, default="/.cache/modelscope/hub/", help="Local dir to save the model")
args = parser.parse_args()

if args.dir is not None:
    args.dir = os.path.join(args.dir, args.model_id)
os.makedirs(args.dir, exist_ok=True)

model = snapshot_download(model_id=args.model_id, cache_dir=args.dir)
```

批量下载示例：
```bash
models=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

for model_name in "${models[@]}"
do
    python modelscope_downloader.py -m "${model_name}"
done
```

## SFT 训练

## PPO 训练

## 奖励模型/功能

在 `verl/verl/utils/reward_score` 文件夹下定义了各种 reward 的计算方式。
