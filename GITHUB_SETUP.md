# GitHub SSH 密钥配置指南

## 🔑 生成的 SSH 公钥

请将以下公钥添加到您的 GitHub 账户：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBMNa6WknaHWHAkeFvhvzXqrgSn0ivgo99VPaI84lvNz ahs-seg@github.com
```

## 📝 添加步骤

1. 访问 GitHub: https://github.com/settings/keys
2. 点击 "New SSH key"
3. Title: `AHS-Seg Server Key`
4. Key: 粘贴上面的公钥内容
5. 点击 "Add SSH key"

## 🔧 验证配置

添加公钥后，在终端运行：

```bash
ssh -T git@github.com
```

如果看到 "Hi ZhaoYi-10-13! You've successfully authenticated..." 表示配置成功。

## 📤 推送更改

配置完成后，运行以下命令推送代码：

```bash
cd /root/AHS-Seg
git add .
git commit -m "Update training results and documentation"
git push origin main
```

## 📊 本次更新内容

- ✅ 完整的训练评估记录 (iter 5K - 35K)
- ✅ 所有 Metrics 统计 (mIoU, fwIoU, mACC, pACC)
- ✅ 更新的配置文件
- ✅ ADE20K-847 数据集准备脚本
- ✅ 增强的模型代码

## 当前训练进度

- **迭代**: 35,339 / 80,000 (44.2%)
- **最新 mIoU**: 42.23% @ iter 35,000
- **预计完成**: 今晚 19:07
