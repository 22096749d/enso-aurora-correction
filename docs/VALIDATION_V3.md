# v3交付验证记录
日期：2026-09-12。

已完成：
- 12项单元/接口测试通过：球面坐标、跨年UTC、ENSO滞后、数据划分、外部预测、文件保护、任务指纹、结果篡改检测。
- 新增交接测试验证：导出排除数据/凭证；全部失败仍能生成诊断ZIP；完整结果验收及被改写后拒绝。
- 合成数据完成baseline、两类有/无ENSO补丁训练、验证、留出测试、无未来标签的apply，SMOKE PASS。

尚未完成：
- Windows真实安装、真实NOAA下载、CDS请求及真实NetCDF转换。
- Aurora权重下载、H800显存/耗时和真实追踪质量。
- 任意真实Aurora预测准确率或ENSO显著增益。

因此服务器首个6小时和120小时任务不可省略。基础权重历史资料重叠也尚未核清。
本包不附合成成绩文件，以免与真实研究结果混淆。

本地代码文件说明：
|文件|作用|
|---|---|
|scripts/pipeline.py|下载、预处理、基线、外部预测导入、两类补丁、最终测试、apply|
|scripts/case_tools.py|无未来标签单例导出与HTML图|
|scripts/common.py|UTC、模型ID、球面预测接口、SHA工具|
|scripts/exchange.py|起报计划、基础结果汇总|
|scripts/server.py|ERA5下载和校验、冻结Aurora、官方Tracker|
|scripts/worker.py|累计批量范围、子进程运行和日志|
|scripts/handoff.py|GitHub白名单导出、环境盘点、完整回传ZIP、本地验收|
|scripts/worker.slurm|同学按学校队列调整的作业模板|

旧scripts/run_server.slurm和旧双端手册仅供历史参考；本版以A/B/C和worker.slurm为准。
