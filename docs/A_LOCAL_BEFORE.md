# A：本地准备（你操作）
所有命令在Windows Anaconda Prompt逐行运行，不是Python >>>或Linux终端。
项目根：F:\ENSO\enso_collab_v3，旧项目不删除。

## 1 解压和安装：建立独立环境
将ENSO_Collaboration_v3.zip解压到F:\ENSO；确认START_HERE.md位于enso_collab_v3内。
已有Conda可直接用；没有从 https://www.anaconda.com/docs/getting-started/miniconda/install 安装。
```bat
cd /d F:\ENSO\enso_collab_v3
conda create -n enso-collab python=3.12 -y
conda activate enso-collab
python -m pip install -r requirements-local.txt
python -c "import numpy,pandas,sklearn; print('LOCAL OK')"
```
目的：隔离旧FNO环境。成功显示LOCAL OK。以后只需activate和cd。
本地训练用CPU，不需要本地CUDA、Torch、Aurora。5070 Ti可用于其他工作。
预留本地10—20GB，Conda缓存所在C盘也留10—20GB；不是说两个CSV需要这么大。

## 2 程序测试：先排除软件问题
```bat
python -m unittest discover -s tests -v
python scripts\pipeline.py smoke --run runs\smoke_v3
```
成功打印OK和SMOKE PASS。smoke是合成数据，不是研究成绩。
已有输出不覆盖；重跑用smoke_v4等新名字。任一步失败先解决，不盲目继续。

## 3 获取真实标签和ENSO
```bat
python scripts\pipeline.py download
```
|文件|来源/内容|用途|
|---|---|---|
|data/raw/ibtracs.WP.list.v04r01.csv|NOAA IBTrACS西北太平洋相关热带气旋最佳路径|历史/当前位置及未来验证真值，固定USA_LAT/USA_LON，不混机构|
|data/raw/nina34.anom.data|NOAA PSL月度Niño3.4海温距平|连续ENSO条件；不是ONI事件分类|

官方地址：
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv
https://psl.noaa.gov/data/correlation/nina34.anom.data
可把旧项目同名原始文件复制到data/raw，不要Excel重存。
下载失败可浏览器另存，检查不是HTML错误页/多了.txt。
prepare记录SHA256和指数页尾版本；冻结样本后不要再更新原始文件。
没有按强度阈值过滤，不能称“仅台风级样本”。

## 4 本地弱基线：先做真实资料的小补丁闭环
```bat
python scripts\pipeline.py prepare --run runs\local_baseline
python scripts\pipeline.py base --run runs\local_baseline
python scripts\pipeline.py train --run runs\local_baseline
python scripts\pipeline.py test --run runs\local_baseline
```
prepare构造样本；base训练简单轨迹预测器；train学习残差；test测留出误差。
查看sample_counts.csv、validation_search.csv、test_metrics.csv。
这不是Aurora，不能把结果当Aurora成绩，也不能直接迁移该补丁到Aurora。
基线期次与Aurora期次不同，见旧手册第5节；此练习可以在等服务器时先完成。

## 5 为Aurora冻结标签样本
```bat
python scripts\pipeline.py prepare --run runs\aurora_samples --config configs\aurora_pilot.json
```
输入当前位置和最近24小时轨迹；标签是24/48/72/96/120小时位置。
ENSO取滞后两个完整月及更早月份/趋势；不用起报月尚未结束的全月平均。
这是修订指数、最佳路径和再分析的回顾性研究，不是历史实时业务重放。

示例划分：补丁训练2016—2020，验证2021—2022，测试2023—2025。
这只代表补丁时间留出；Aurora基础权重训练/模型选择重叠尚未核清。
configs/backbone_audit.json有未知项，正式结论前需要核实并冻结期次。
检查sample_counts.csv各集合和时效是否有数据。长时效缺真值会减少样本。

## 6 生成小任务：先测一次真实Aurora推理
```bat
python scripts\exchange.py plan --run runs\aurora_samples --directory jobs\pilot --max-per-split 2
python scripts\exchange.py verify-plan --directory jobs\pilot
```
最多6个台风/起报案例，不能正式训练。服务器先跑最早1个，再完成pilot。
同一stamp的多个台风共享全球推理，计算预算看独立stamp数。
plan.csv仅SID、UTC起报、当前位置、split，无未来标签/ENSO。
manifest锁计划和本地samples指纹。不要用Excel改写。

## 7 导出要上传GitHub的目录
```bat
python scripts\handoff.py export --job pilot --destination github_export_pilot
```
只上传F:\ENSO\enso_collab_v3\github_export_pilot。
白名单包含代码、依赖、配置、说明和jobs/pilot；不会复制原始数据、权重、结果、凭证。
检查导出目录没有data/models/runs和.env/.cdsapirc。

### GitHub Desktop方式
1. 从 https://desktop.github.com/ 安装，登录自己的GitHub账号。
2. File → Add Local Repository，选github_export_pilot。
3. 提示非仓库时选择创建，确认位置仍是这个已有目录，不新套一层空目录。
4. 检查Changes，应有scripts/docs/configs/jobs，无数据、权重、密钥。
5. Summary填Initial Aurora ENSO pilot，Commit to main。
6. Publish repository，建议先保持private，名称例如enso-aurora-correction。
7. GitHub网页Settings → Collaborators邀请同学。对方接受后可clone。
8. 发仓库网址、commit和docs/B_SERVER_COLLEAGUE.md，说明首轮只跑pilot。
不用共享学校账号。本交付未代你创建远程GitHub仓库。

### 已装Git也可命令行
先在GitHub网页建立空仓库（不自动生成README）。YOUR_ACCOUNT换实际用户名：
```bat
cd /d F:\ENSO\enso_collab_v3\github_export_pilot
git init
git add .
git status
git commit -m "Initial Aurora ENSO pilot"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/enso-aurora-correction.git
git push -u origin main
cd /d F:\ENSO\enso_collab_v3
```
认证用Git官方登录界面，不把token写进URL。

## 8 pilot回传通过后再扩大
先做C1—C2，确认耗时、追踪质量及设计，再生成每集合最多100案例的探索批次：
```bat
python scripts\exchange.py plan --run runs\aurora_samples --directory jobs\batch --max-per-split 100
python scripts\handoff.py export --job batch --destination github_export_batch
```
100不是统计充分性承诺。抽样按每集合时间均匀取，不保证ENSO事件/年份均衡，要检查覆盖。
全取用--max-per-split 0，另取job名full；先按独立起报数估成本。
把github_export_batch/jobs/batch复制到已存在的GitHub本地仓库jobs/batch，然后commit/push。
代码未改时只提交新任务；代码改动需同步并冻结新版本。不要在服务器运行中更新代码。
同学按B7运行。pilot与batch分目录，重叠预测不要合并重复计数。
正式样本计划须在看测试成绩前固定，不能挑容易改善的台风。

