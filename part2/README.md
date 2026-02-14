# GreenEdge MVP — Part 2 (RL Edge Placement)

## Amaç
Trafik isteklerini (latency + energy maliyeti) minimize edecek şekilde **edge-a / edge-b / cloud** arasında yönlendiren
bir **Reinforcement Learning (RL)** ajanı geliştirmek.

## Çıktı (Gösterim)
- Basit bir simülatör: istekler gelir, her hedefin gecikme/enerji maliyeti vardır.
- RL ajanı karar verir (action): edge-a / edge-b / cloud
- Metrikler: p95 latency, energy/Mbps, SLA ihlali, toplam ödül
- Küçük bir arayüz (web veya local) ile “hangi kararlar alındı?” gösterimi
