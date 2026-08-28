# gem5 branch backup bundles

`public` 远端拒绝完整 gem5 历史推送（上游 blob:none partial clone 缺对象）。
这里存放**有界 bundle**：仅含本分支自有提交的对象，基线 commit 作为
prerequisite 记录在 bundle 头里。恢复方式：

    git -C projects/gem5 bundle verify <bundle>   # 需先有基线 a6e9be550
    git -C projects/gem5 fetch <bundle> zcode/gem5-hybrid-cta2:zcode/gem5-hybrid-cta2

- `zcode-gem5-hybrid-cta2.bundle`: hybrid CTA 重实现（2 commits:
  88162bb3f executor 重实现+验收，4bd5e5a75 idle park 修复 DRAMPower 燃烧）。
  基线 = zcode/gem5-rebuild 顶 a6e9be550（tag gem5-verified-baseline-20260826）。
