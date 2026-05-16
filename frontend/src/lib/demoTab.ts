import type { GeneratedTab } from "../types/tab";

export const demoTab: GeneratedTab = {
  id: "demo",
  title: "TaBee Demo",
  fileName: "Built-in AlphaTex sample",
  instrument: "bass",
  tempo: 98,
  createdAt: "Demo",
  alphaTex: String.raw`\title "TaBee Demo"
\artist "Generated Bass"
\track "Bass"
\staff {tabs}
\tuning (G2 D2 A1 E1)
:8 0.4 2.4 3.4 5.4 0.3 2.3 3.3 5.3 |
:8 7.3 5.3 3.3 2.3 5.4 3.4 2.4 0.4 |
:8 0.4 0.4 3.4 5.4 0.3 0.3 3.3 5.3 |
:8 7.3 7.3 5.3 3.3 2.3 0.3 3.4 5.4 |
:4 0.4 3.4 5.4 0.3 |
:8 2.3 3.3 5.3 7.3 5.3 3.3 2.3 0.3 |`
};
