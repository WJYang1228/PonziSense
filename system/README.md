# 原型系统（非论文开源部分）

本目录为**工程化 Web 工作台**：Flask 应用、模板、静态资源，以及调用根目录算法的桥接层（`bootstrap.py`、`webapp/services/`）。

公开论文/代码仓库时，可**删除本目录**或加入 `.gitignore`，只保留根目录的训练、评估与 `predict.py` 推理接口。

## 依赖

与仓库根目录相同，并需安装 Flask（见根目录 `requirements.txt`）。

## 运行前准备

1. 在仓库根目录完成训练，得到 `outputs/checkpoints/best_model.pt`（路径规则与根目录 `utils/paths.py`、`predict.py` 一致）。
2. 确保可联网下载 GraphCodeBERT（首次推理），或已配置本地 Hugging Face 缓存。

## 启动方式

**必须在仓库根目录**（包含 `predict.py`、`models/` 的目录）执行，以便 `bootstrap` 将根目录加入 `sys.path`：

```bash
cd /path/to/ponzi_exp
python -m system
```

或仓库根目录下：

```bash
./run_web.sh
```

浏览器访问：`http://127.0.0.1:7860/`

- 工作台：`/`
- 分析页：`/workspace/analysis`
- 健康检查：`/system/health`
- 关于页：`/system/about`

环境变量（可选）：

- `PONZI_HOST`、`PONZI_PORT`：监听地址与端口  
- `PONZI_CKPT`：覆盖权重文件路径  
- `APP_PRODUCT_NAME`、`APP_VERSION` 等：见 `system/webapp/config.py`

## 生产部署（示例）

```bash
gunicorn -w 2 -b 0.0.0.0:7860 'system.app:app'
```

（需自行 `pip install gunicorn`，工作目录仍为仓库根目录。）

## 与「算法开源」的边界

| 位置 | 说明 |
|------|------|
| 仓库根目录 `train.py`、`evaluate.py`、`predict.py`、`models/`、`data/`、`configs/` 等 | 论文复现与算法能力验证 |
| `system/` | 内部原型 UI，可不随论文公开 |

算法能力验证不必依赖本目录：使用根目录 `python evaluate.py`、`python predict.py` 即可；`system` 仅提供可视化与产品化演示。
