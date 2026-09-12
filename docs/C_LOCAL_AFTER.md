# C：收到结果后本地处理
Windows Anaconda Prompt：
```bat
conda activate enso-collab
cd /d F:\ENSO\enso_collab_v3
```

## 1 pilot解压验收：检查对应任务与文件完整性
保存pilot_v1.zip，解压到received/pilot_v1，确认bundle_manifest.json直接在其中。
```bat
python scripts\handoff.py accept --job pilot --directory received\pilot_v1 --run runs\aurora_samples
```
验证整包SHA、原plan、原samples、模型ID、20步及坐标/时效，打印覆盖。
未运行的全局起报会被拒绝；部分Tracker缺失须看diagnostics和有效时效数。
diagnostics_only不能训练，带错误原文和日志反馈，不绕过校验。
验收不自动判断是否追错台风或科学有效。

## 2 小试只导入，不训练
```bat
python scripts\pipeline.py prepare --run runs\pilot_import --config configs\aurora_pilot.json
python scripts\pipeline.py external --run runs\pilot_import --config configs\aurora_pilot.json --input received\pilot_v1\result\forecasts.csv --model-id aurora025_pretrained_era5_officialtracker
```
external_coverage相对全部本地候选样本计算匹配，pilot低覆盖正常；
返回coverage相对真正派发任务计算完成度，两者不同。
最多6案例不够正式训练。看单例wall_seconds×后续独立起报数估算GPU时间，另计下载/排队。
人工看路径和失败。若根据test案例调Tracker，该部分已参与开发，须重新安排外层测试。
确认后回A8生成batch，让同学按B7执行。

## 3 批量验收
将batch_v1.zip解压到received/batch_v1：
```bat
python scripts\handoff.py accept --job batch --directory received\batch_v1 --run runs\aurora_samples
```
不要把重叠pilot预测再次合并。全局失败先补跑；追踪失败不能静默删除。
报告每时效有效样本和失败/回退。消散导致标签不足也需说明：误差是在有真值和有效输出条件下评价。
覆盖严重偏向某年或气候背景，先解决数据/设计问题。

## 4 训练Aurora两类补丁
```bat
python scripts\pipeline.py prepare --run runs\aurora_fit --config configs\aurora_pilot.json
python scripts\pipeline.py external --run runs\aurora_fit --config configs\aurora_pilot.json --input received\batch_v1\result\forecasts.csv --model-id aurora025_pretrained_era5_officialtracker
python scripts\pipeline.py train --run runs\aurora_fit --config configs\aurora_pilot.json
```
这里不执行base，服务器返回的Aurora就是冻结基础模型。原始data/raw保持不变。
r=真实位移−原预测位移；补丁g学习r，输出原预测+αg，局部球面坐标以公里表示再还原经纬度。
特征：当前位置、最近运动、月份、原预测；ENSO版另加滞后指数与趋势。
每个时效独立训练，超参数及修正幅度α只在验证集选。
|对象|目的|
|---|---|
|base|Aurora原始参考|
|mean|最简单平均偏差修正|
|ridge_noenso、ridge_enso|岭回归无/有ENSO|
|tree_noenso、tree_enso|梯度提升树无/有ENSO|
CPU足够，这不是Aurora微调/LoRA。产物patch_models.joblib和validation_search.csv。
α=0表示验证不支持修正，应保留。20行最低门槛不代表独立年份/台风/事件数充分。
出现报错停在该步；新run重做，不覆盖旧实验。

## 5 最终测试
冻结设计后：
```bat
python scripts\pipeline.py test --run runs\aurora_fit --config configs\aurora_pilot.json
python -m pip freeze > runs\aurora_fit\environment.txt
```
|文件|读法|
|---|---|
|test_metrics.csv|各时效mean_km、median_km、p90_km及样本数，越小越好|
|test_predictions.csv|每案例每方法预测，便于核查|
|paired_year_bootstrap.csv|gain_km正数是改善，配对年度重抽样区间|
核心比较是有ENSO对同类无ENSO，不只是对base；否则无法分离常规后处理增益。
3年测试的年度区间不稳健；月度暖冷分箱不是官方ONI事件。
基础权重历史重叠未核清，不能宣称完全独立于基础模型的泛化。
测试后改设计需新的外层评价，不继续称未见测试。
本包计算大圆距离，没有实现登陆点、概率评分、因果归因。

## 6 单例接口与图
```bat
python scripts\case_tools.py export --run runs\aurora_fit --output received\case_input.csv
python scripts\pipeline.py apply --run runs\aurora_fit --input received\case_input.csv --output runs\aurora_fit\case_corrected.csv --method ridge_enso
python scripts\case_tools.py plot --run runs\aurora_fit --output runs\aurora_fit\case_plot.html
```
export选五时效完整的历史案例并移除未来标签；apply输出corrected_lat/lon。
plot展示已保存test预测，不是直接读取case_corrected.csv的实时地图；HTML无海岸线。
这展示“原路径后挂补丁”的接口，不是业务实时观测平台。
真实新台风仍需当时可得的定位、历史和ENSO。

## 7 汇报与保存
保存原始数据指纹、samples、计划、Git commit、服务器ZIP、配置、小模型及所有评价。
向教授报告期次和样本数、原模型版本、失败/回退、两组ENSO增益、年度区间和案例。
不能用合成smoke或本地弱基线代替Aurora实测；结果不改善也保留。

