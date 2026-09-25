"""
Qwen3-VL-Embedding encoder — HTTP client 模式。

连接独立 embedding server 进程（embedding_server.py）来编码 text/image。
避免 Ray actor (AgentLoopWorker) 中 CUDA 不可用的问题。

Ray actor 不分配 GPU (num_gpus=0)，且 CUDA context 可能因 fork 损坏，
导致 torch.cuda.set_device(0) 报 "No CUDA GPUs are available"。
独立 server 进程的 CUDA 完全正常，通过 HTTP 提供编码服务。

接口和之前完全一致（encode_text / encode_image / dim / get_model），
retriever 无需修改。
"""
import os
import json
import base64
import urllib.request
import numpy as np

# embedding server 地址（可通过环境变量覆盖）
_SERVER_URL = os.environ.get('EMBEDDING_SERVER_URL', 'http://localhost:9981')
_TIMEOUT = 120  # 单次请求超时（秒）
_RETRIES = 3    # 重试次数

# instruction 选择：实测 "Represent the text for retrieval." 在我们的 HTTP server
# (bfloat16+eager) 环境下整体效果更好（baseline 0.645 vs 0.605）。
# LTMemory 用 default_instruction="Represent the user's input." 但它是进程内 float32。
# 两者 dtype/attn 不同导致 instruction 交互不同，直接搬 instruction 反而更差。
# 环境变量 EMB_TEXT_INSTR 可覆盖（方案 F: 配合 9982 float32 server 完全对齐 LTMemory）
_text_instr = os.environ.get('EMB_TEXT_INSTR', "Represent the text for retrieval.")
_img_instr = os.environ.get('EMB_IMG_INSTR', "Represent the image for retrieval.")


def _post(req: dict, timeout: int = None) -> np.ndarray:
    """发送编码请求到 server，返回 numpy embedding。"""
    timeout = timeout or _TIMEOUT
    last_err = None
    for attempt in range(_RETRIES):
        try:
            body = json.dumps(req).encode()
            r = urllib.request.Request(
                _SERVER_URL, data=body,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                result = json.loads(resp.read())
                b64 = result['emb']
                shape = tuple(result['shape'])
                buf = base64.b64decode(b64)
                return np.frombuffer(buf, dtype=np.float32).reshape(shape)
        except Exception as e:
            last_err = e
            if attempt < _RETRIES - 1:
                import time
                time.sleep(1)
    raise RuntimeError(f"Embedding server request failed ({_RETRIES} retries): {last_err}")


def encode_text(text: str | list[str], instruction: str = None) -> np.ndarray:
    """编码文本，返回 (dim,) 或 (n, dim) numpy。"""
    instr = instruction or _text_instr
    if isinstance(text, str):
        return _post({'type': 'text', 'text': text, 'instr': instr})
    else:
        # batch 请求（减少 HTTP 往返）
        return _post({'type': 'batch_text', 'texts': list(text), 'instr': instr})


def encode_image(image_path: str | list[str], instruction: str = None) -> np.ndarray:
    """编码图片，返回 (dim,) 或 (n, dim) numpy。text+image 同空间。"""
    instr = instruction or _img_instr
    if isinstance(image_path, str):
        return _post({'type': 'image', 'image': image_path, 'instr': instr}, timeout=180)
    else:
        return _post({'type': 'batch_image', 'images': list(image_path), 'instr': instr}, timeout=300)


# 兼容旧接口名
encode = encode_text
encode_text_clip = encode_text


def dim() -> int:
    return 2048


def get_model():
    """No-op — 模型在 server 进程中。保留接口兼容。"""
    return None
