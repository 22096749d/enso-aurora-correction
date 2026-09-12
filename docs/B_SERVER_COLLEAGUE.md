# B：给同学的Linux服务器代跑指南
只需执行这份。无需为用户开学校账号。不训练/微调Aurora，只做冻结推理和路径追踪。
首轮最多6案例，先1个；批量规模在实测单例成本后约定。

## 0 固定模型与数据
已知你有H800 80GB且用过ERA5/HRES T0，尚未确认具体环境、权重、数据年份。
本版：AuroraPretrained + aurora-0.25-pretrained.ckpt + ERA5全球0.25°。
官方ERA5推荐匹配：https://microsoft.github.io/aurora/models.html
官方追踪示例：https://microsoft.github.io/aurora/example_tc_tracking.html
Fine-Tuned配HRES T0是另一条路线，不直接改权重名或换输入。
已有ERA5需先验证变量/层数/时刻/单位和来源；本包不猜测任意本地目录。
最直接的首例是CDS下载小量初值。若希望复用既有文件，先提供变量维度清单再做版本化适配。
不要伪造下载manifest来让任意文件通过检查。

## 1 获取代码
在学校允许的工作目录，替换真实仓库URL：
```bash
git clone https://github.com/YOUR_ACCOUNT/enso-aurora-correction.git
cd enso-aurora-correction
git rev-parse HEAD
```
与用户提供commit一致。私有仓库使用自己GitHub认证，token不进URL。
后续接batch用git pull；保留本地修改并先协调，勿强制覆盖。

## 2 环境盘点和部署
已有numpy/pandas的环境先只读检查：
```bash
python scripts/handoff.py inventory --destination logs/preflight.json
```
可先发preflight.json给用户，不加载大模型、不读凭证。
本版锁microsoft-aurora==2.0.1，它是软件包版本，不是模型版本。
若已有环境不匹配，建立独立环境，不升级别人现用环境：
```bash
conda create -n enso-aurora-v3 python=3.12 -y
conda activate enso-aurora-v3
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server.txt
python -m pip check
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
python scripts/handoff.py inventory --destination logs/environment_ready.json
```
配方待实机验证，驱动支持以学校配置为准，不sudo改共享驱动。
完全匹配可复用现有环境。安装失败返回错误，不--no-deps硬装。
建议单卡80GB、CPU8核、RAM64GB起；小试磁盘50—100GB。
每起报两个全球时刻的原始float32数组约0.57GB；实际压缩缓存需测量。
批量可能数百GB，不先下载整个多年ERA5全场。

## 3 锁权重并准备最早一个起报
```bash
python scripts/exchange.py verify-plan --directory jobs/pilot
python scripts/server.py weights
```
下载指定checkpoint并保存models/checkpoint.json的revision/hash。
默认缓存models/hf_cache；如已有匹配HF缓存，可预先设置HF_HOME，仍由程序锁定权重。
不能使用不同模型冒充相同ID。

CDS认证由你按官方 https://cds.climate.copernicus.eu/how-to-api 配置并接受ERA5两个数据集条款。
认证不进仓库/返回包。
```bash
python scripts/worker.py data --job pilot --through 1
```
through=1是最早一个独立UTC起报。取t-6h和t，跨日分请求；静态字段取固定一份。
输出data/era5/YYYYMMDDHH/surface.nc、upper.nc及manifest、static.nc。
地面t2m/u10/v10/msl，高空t/u/v/q/z，13层50—1000hPa，静态lsm/slt/z。
程序检查全球721×1440、时间、有限数值、Pa和位势单位。
.part残片导致停止；检查并归档具体残片，不删除整个数据目录。

## 4 GPU单步检查
仅在已分配的GPU计算节点执行，不能在集群登录节点运行：
```bash
python scripts/server.py check
python scripts/server.py run --plan jobs/pilot/plan.csv --output server_runs/pilot_step1 --limit 1 --steps 1 --save-msl
```
成功标志done.json，输出6小时预测、峰值显存及追踪情况。
天气模型成功不代表Tracker成功，要看tracker_errors.json。
遇到不兼容先把错误发回。本包未在该H800真实端到端验证，此步是必要验收。

## 5 跑120小时和完整pilot
```bash
python scripts/worker.py run --job pilot --through 1
```
自动20×6小时，输出server_runs/pilot；日志自动保存logs/pilot。
检查done.json的wall_seconds、峰值、追踪结果，正常后：
```bash
python scripts/worker.py data --job pilot --through 6
python scripts/worker.py run --job pilot --through 6
```
through为累计范围，超过计划长度自动截断；已有连续完成前缀跳过。
半成品/error目录会停止；保留错误并定位后归档那个具体目录再重试。
每个job仅一个进程运行，无自动多GPU调度。

### Slurm方式
scripts/worker.slurm需按学校填写partition/account/GPU约束和Conda初始化。
模板不能保证队列正确；先下载初值，再提交：
```bash
mkdir -p logs
sbatch --export=ALL,ENSO_JOB=pilot,ENSO_THROUGH=1 scripts/worker.slurm
```
首个完成后再ENSO_THROUGH=6，不同时提交同一job。
首次6小时检查应在学校提供的交互GPU申请内运行。

## 6 打包回传：正常和失败都能交接
```bash
python scripts/handoff.py bundle --job pilot --output server_runs/pilot --destination returns/pilot_v1
```
只需发returns/pilot_v1.zip，不必推GitHub、不发权重或ERA5。
先查看日志是否含不宜外发信息，需要脱敏则记录并重建包，勿改已校验文件。
|包内文件|用途|
|---|---|
|result/forecasts.csv|SID、起报、lead_h、原始预测经纬度、Tracker回退|
|result/coverage.csv|所有计划案例的运行状态和有效时效数|
|result/return_manifest.json|模型和软件身份、结果指纹|
|plan.csv、manifest.json|用户原任务|
|diagnostics/各时刻/done.json或error.json|耗时、显存、输入指纹或失败|
|diagnostics/各时刻/tracker_errors.json|追踪错误详情|
|environment.json、checkpoint.json|commit、软件、权重版本|
|logs/|本job自动记录的输出/错误|
|bundle_manifest.json|整包内容指纹|

全部失败也能生成diagnostics_only包，不能训练但可以排错。
若安装前失败，另发preflight和安装错误文本。不要发凭证。
旧返回包不覆盖，更新用pilot_v2。单步诊断天气场不默认回传，必要时另商量。

## 7 批量任务
用户已确认pilot并推送jobs/batch后，git pull、核对版本：
```bash
python scripts/worker.py data --job batch --through 10
python scripts/worker.py run --job batch --through 10
```
之后20、30……累计推进，按实际时限决定增量。data和run的through含义一致。
完成：
```bash
python scripts/handoff.py bundle --job batch --output server_runs/batch --destination returns/batch_v1
```
发送batch_v1.zip。未完成可先返回进度，但本地默认拒绝拿未执行完的计划直接训练。
保留权重锁、初值与指纹、代码commit及运行记录，双方约定存储保留期限。

