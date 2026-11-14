# 迁移状态

目标：`kkkkkkkee/CoT-length-generalization`。

当前：本地准备中，**尚未确认 Git push 或 Release 上传成功**。恢复命令仅在附件发布完成后可用。文件清单见 `archive-manifest.json`（生成完成后存在）。

Release tag：`experiment-archive-2026-11-14`。完整 `runs/` 按最多 512 MiB 原始内容分包，逐包 gzip 压缩。每包上传后重新下载验证，全部完成才发布 Release。旧服务器原始文件不会删除。

迁移工具已用临时样本验证：打包/恢复一致、重复恢复、拒绝覆盖不同文件、拒绝目录穿越。
