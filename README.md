# CoT 长度泛化：复现与机制实验

研究 *Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization* 的独立实证复现，以及结构约束、位置偏置和注意力因果机制。

两类模型必须区分：早期及第二轮复现使用端到端双头模型；Phase C/D、位置与注意力机制实验使用固定 one-hot、单头、两个可训练打分块的分阶段模型。两类模型都不等同于原文定理全部特殊假设的严格实现。

## 先读这些文件

- [第二轮双头复现结果](notes/replication-results-2026-11-12.md)：最新完成的 39 个训练运行/阶段，51 个终点评测。
- [第二轮预案](notes/replication-plan-2026-11-12.md)。默认尺度下 S5 自训练 L160 的 greedy value 整链均值 93.82%；C6 最好基线 52.08%，尚未完整重现原文的极长链表现。
- [30 分钟机制实验报告](reports/experiment-report-2026-11-11/实验报告-30分钟版.md)：早于第二轮复现，不包含 11 月 12 日的新结果。
- [完整机制实验报告](reports/experiment-report-2026-11-11/实验总报告.md)。
- [原文精读](notes/paper-review-zh.md)。

## 服务器迁移

代码、测试、报告、图表和实验预案放在 Git 仓库；体积大的 `runs/` 使用 GitHub Release 分包保存。**仅 git clone 不能取得模型和原始结果，必须完成附件恢复。** 上传状态见 [迁移状态](migration/STATUS.md)。

```bash
git clone https://github.com/kkkkkkkee/CoT-length-generalization.git
cd CoT-length-generalization
gh auth login --hostname github.com --git-protocol https --web
python migration/github_archive.py restore \
  --repo kkkkkkkee/CoT-length-generalization \
  --tag experiment-archive-2026-11-14 --root "$PWD"
```

脚本逐包下载、验证压缩包 SHA-256、恢复并核对每个文件的 SHA-256。可重复执行；内容相同的文件跳过，不同的现有文件拒绝覆盖。完整实验记录约 40 GB，建议至少预留 50 GB。仅恢复数据不需要 PyTorch。

历史配置与 checkpoint 中含旧服务器绝对路径 `/root/shared-nvme/cot-length-generalization`。为了原样恢复旧运行，最简单是克隆到同一路径，或在新服务器有权限时将该路径链接至新克隆目录。不要批量修改历史 JSON/检查点：这会改变原始哈希。独立启动新的实验可使用代码根据文件位置推导的项目路径，但恢复旧运行前必须检查其绝对路径引用。

## 环境

旧环境：Python 3.12.3，NVIDIA RTX 4090，PyTorch `2.7.0a0+7c8ec84dab.nv25.03`（CUDA 12.8），NumPy 1.26.4，Matplotlib 3.10.1，Pillow 11.1.0，pytest 8.1.1。这是 NVIDIA 构建的预发布 PyTorch，不应声称普通 PyPI 的 2.7 与其逐比特相同。

新服务器先按显卡/驱动安装兼容 CUDA 的 PyTorch，再安装其余 Python 依赖。不同版本需重跑测试和工程 smoke，不能直接假定结果完全相同。

```bash
python -m pip install -r requirements-analysis.txt
python -m pytest -q tests
```

中文图表需系统 CJK 字体，例如 Noto Sans CJK。

## 重新运行

最新正式实验已完成。需要恢复其意外中断运行时，保留路径映射并调用冻结源码：

```bash
python runs/replication-2026-11-12/source/run_replication_campaign.py \
  --frozen --output runs/replication-2026-11-12
```

若要开展新的独立运行，请选新的输出目录：

```bash
python run_replication_campaign.py --smoke --output runs/new-server-smoke
python run_replication_campaign.py --output runs/new-replication
```

其他实验的启动入口为项目根目录 `run_*.py`，设计与历史说明在 `notes/`，各运行冻结源码和参数在 `runs/`。部分历史驱动有固定日期目录，重新运行前先审阅路径，避免误用已完成结果。

## 存档原则

存档包括全部历史 `runs/` 的训练数据、伪标签、初始/中间/最终检查点、可用的 Adam/RNG 断点状态、评测、日志及冻结源码；仅排除 Python 缓存和瞬时文件锁。保留中间检查点是为了支持已有训练动力学分析。早期训练脚本未保存 Adam 状态，其检查点可做评测或新阶段初始化，不能冒充精确续训状态。

原服务器 SSH 密钥、GitHub 凭据、环境变量和项目外个人文件不会纳入存档。
