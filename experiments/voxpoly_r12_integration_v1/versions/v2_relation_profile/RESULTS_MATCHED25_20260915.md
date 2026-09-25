# Audio relation/profile v2：matched-25 正式验收

状态：**完成，四类预注册门槛全部通过**。本次只比较冻结的 current R12
与 v2；评测后没有重跑、调参或改动检索结果。

## 结论

在同一批 25 道题、同一 `gpt-4.1-mini`、同一 answer/judge prompt、同一
冻结 route、同一 Top-30 和同一音频推断 runtime character 下，v2 的
LLM-score 从 R12 的 **0.7900** 提升到 **0.9000**，Acc@0.75 从
**0.8000** 提升到 **0.9600**。gold evidence recall@30 从 **0.7133**
提升到 **0.9400**。

| 四大类 | QA | old 参考 | current R12 | v2 | R12 recall | v2 recall | 验收 |
|---|---:|---:|---:|---:|---:|---:|---|
| Persona | 2 | 0.7500 | 0.6250 | **0.8750** | 0.5000 | **1.0000** | 通过：v2 ≥ 0.75 |
| Attribution | 12 | 0.7708 | 0.6667 | **0.8542** | 0.4861 | **0.8750** | 通过：v2 ≥ 0.7708 |
| Retrieval reasoning | 7 | 0.8214 | **0.9286** | **0.9286** | 1.0000 | 1.0000 | 通过：保持 R12 且高于 old |
| Memory evolution/conflict | 4 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 通过：保持 1.0 |
| **整体** | **25** | **0.8200** | **0.7900** | **0.9000** | **0.7133** | **0.9400** | **全部通过** |

这里的 old 是按完全相同的 case/QA ID 对齐的历史 `gpt-4.1-mini` 结果，
只用于诊断和门槛参考，不是正式 matched 的第三臂。它使用 Velen、历史
answer context/pipeline，而且 Persona 使用了 oracle-derived asker；正式因果比较
只能看本次 AIGC 运行中的 current R12 与 v2。

另需注意 Persona 只有 2 道题；它通过了本次冻结面板的门槛，但不能单独支持
稳健的跨 case 泛化结论。

## 精确对齐与胜负

- v2 对 current R12，LLM-score：**4 胜 / 20 平 / 1 负**；总分净增
  2.75，即平均增加 0.11。
- v2 对 current R12，Acc@0.75：**4 胜 / 21 平 / 0 负**。唯一分数下降题
  仍为 0.75，未跨过正确阈值。
- v2 对 current R12，gold recall：**11 胜 / 14 平 / 0 负**。
- v2 对 old，LLM-score：6 胜 / 16 平 / 3 负。R12 对 old：3 胜 / 18 平 /
  4 负。

| Case | QA | 类别 | old | R12 | v2 | R12 recall | v2 recall | v2 vs R12 |
|---|---|---|---:|---:|---:|---:|---:|---|
| G014 | MULTI_003 | Retrieval | 0.50 | 0.75 | 0.75 | 1.000 | 1.000 | 平 |
| G014 | CONFLICT_007 | Evolution/conflict | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G014 | ADVERSARIAL_003 | Retrieval | 0.75 | 0.75 | 0.75 | 1.000 | 1.000 | 平 |
| G014 | PERSONA_004 | Persona | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G014 | PERSONA_005 | Persona | 0.50 | 0.25 | **0.75** | 0.000 | **1.000** | 胜 |
| G014 | ATTRIBUTION_005 | Attribution | 1.00 | 0.00 | 0.00 | 0.167 | **0.833** | 平 |
| G014 | ATTRIBUTION_019 | Attribution | 0.00 | 0.00 | **1.00** | 0.667 | **0.833** | 胜 |
| G014 | ATTRIBUTION_009 | Attribution | 1.00 | 1.00 | 1.00 | 0.333 | **0.833** | 平 |
| G014 | ATTRIBUTION_012 | Attribution | 0.25 | 0.25 | **0.75** | 0.333 | **0.833** | 胜 |
| G014 | CONFLICT_001 | Evolution/conflict | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G014 | SINGLE_008 | Retrieval | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G014 | ATTRIBUTION_001 | Attribution | 1.00 | **1.00** | 0.75 | 0.333 | **0.833** | 负 |
| G014 | ATTRIBUTION_017 | Attribution | 0.00 | 1.00 | 1.00 | 0.333 | **0.833** | 平 |
| G014 | MULTI_001 | Retrieval | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G014 | ATTRIBUTION_015 | Attribution | 1.00 | 1.00 | 1.00 | 0.333 | **0.833** | 平 |
| G015 | ATTRIBUTION_011 | Attribution | 1.00 | 0.75 | 0.75 | 0.333 | **0.833** | 平 |
| G015 | ADVERSARIAL_007 | Retrieval | 0.50 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G015 | ATTRIBUTION_010 | Attribution | 1.00 | 0.00 | **1.00** | 0.500 | **0.833** | 胜 |
| G015 | CONFLICT_002 | Evolution/conflict | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G015 | MULTI_006 | Retrieval | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G019 | ATTRIBUTION_006 | Attribution | 1.00 | 1.00 | 1.00 | 0.500 | **1.000** | 平 |
| G019 | ATTRIBUTION_011 | Attribution | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G019 | CONFLICT_001 | Evolution/conflict | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G019 | TEMPORAL_001 | Retrieval | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |
| G019 | ATTRIBUTION_012 | Attribution | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 平 |

## 两道重点归因题

### ATTRIBUTION_005：recall 已修复，但答案合成仍失败

问题问 Mia 的 “where and what time we're booked” 是对谁说的，gold 为
Priya Shah。R12 与 v2 都回答 Rowan，均得 0；但 recall 已从 1/6 提升到
5/6。v2 把锚点 `S4_T010` 放在第 3 位，把 Priya 的直接回复 `S4_T011`
放在第 4 位，并带有内部 `replied_by` 选择轨迹，另召回相邻的
`S4_T009/S4_T012/S4_T008`，但仍缺 `S4_T007`。更关键的是，排在最前的
`S4_T040/S4_T041` 是另一组高度相似的 “Mia 问 Rowan / Rowan 回复” 干扰对；
最终 raw context 又没有把 reply edge 显式呈现给 answer prompt。因此这是
“主要判别证据已召回，但关系呈现与答案合成仍失败”，而不是简单的直接回复
缺失。按冻结约束，本轮不为这一题追加规则或重跑。

### ATTRIBUTION_001：语义正确，0.25 下降来自 judge 精度敏感

gold 为 Rowan。R12 和 v2 的答案都正确说 Rowan，并都额外给出同一个日期
2026-06-01；文本差异主要是引号样式。R12 judge 给 1.0，v2 judge 因“日期
不是 Ground Truth 所需”给 0.75。与此同时 recall 从 2/6 提升到 5/6，v2
已包含问题锚点、Rowan 的直接回答以及同 session 邻域。这更符合 judge
不一致/措辞敏感，而不是语义错误或检索回退；Acc@0.75 仍为正确。

## 身份、盲测与成本

- Persona 正式臂从 query waveform 经 ECAPA 和冻结 EMA registry 得到
  `asker_ref`；没有把文件名或旧 `asker_name` 输入检索器。旧姓名只用于
  推断后的离线等价核对。
- retrieval 函数不接受 benchmark label、QA type、gold evidence 或答案。
  所有检索上下文在读取 gold/answer 前冻结并哈希。
- provider：AIGC native；model：`gpt-4.1-mini`；Top-K：30；Velen fallback
  关闭。两个已配置 AIGC key 轮换，发生 1 次 429 后成功切换；无失败题。
- 实际用量：75 calls，92,879 prompt tokens，3,140 completion tokens，
  **96,019 total tokens**。AIGC 等价额度成本 **CNY 0.3090**；Velen 调用 0，
  Velen 实付 **CNY 0**。

## 可复现文件

- 正式结果：`artifacts/matched25_eval/matched_pairs.json`
  (`3d76920e6a497c098a563cc761959a5aca941dd4aa40fce9b4fe8ce78c0f6e73`)
- gold 前冻结上下文：`artifacts/matched25_eval/frozen_contexts_before_gold.json`
  (`90d1b704e295b418150dccd9c6104cfe6a8ff7c1208e6d16318e033acac61caf`)
- 调用缓存：`artifacts/matched25_eval/llm_call_cache.json`
  (`4effce98a86436792e6e1892cbef21783615b289cbbc72f4613af75352b0fca2`)
- 运行日志：`artifacts/matched25_eval/run_aigc_gpt41mini.log`
  (`eca3a462d83ad0d403dc1735d6ee462e8295af33ff680cf696bc48b50c2a48ac`)
- retrieval 审计：`artifacts/retrieval_audit.json`
  (`32bcb268bddee85121f8ab256ac5ea1de836e6e8861d12860186d0ac1ff63e60`)
- 机器可读汇总：`artifacts/matched25_eval/summary.json`，由
  `summarize_matched25.py` 仅用本地完整结果确定性生成，不调用 API。

最终完整文件哈希以 `MANIFEST.sha256` 为准。
