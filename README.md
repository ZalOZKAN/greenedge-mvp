# GreenEdge-5G

5G şebekelerinde enerji ve gecikmeyi birlikte optimize eden yapay zekâ tabanlı trafik yönetim sistemi.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Hızlı Bakış

GreenEdge-5G, 5G altyapılarında iş yükü yönlendirme kararlarını PPO (Proximal Policy Optimization) algoritması ile veren bir karar motorudur.

**Temel Bileşenler:**
- Deep Reinforcement Learning (PPO)
- Cloud-Native mimari (Docker/k3s)
- REST API (FastAPI)
- Gerçek zamanlı KPI görselleştirme (Streamlit)

---

## Kurulum ve Çalıştırma

```powershell
# Repository'yi klonla ve ortamı hazırla
git clone https://github.com/your-org/greenedge-mvp.git
cd greenedge-mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Model eğitimi
python -m greenedge.rl.train --algo ppo --steps 20000

# Model değerlendirmesi
python -m greenedge.rl.evaluate --episodes 200

# API sunucusu
python -m greenedge.api.main

# Dashboard
streamlit run greenedge/dashboard/app.py
```

### Test ve Docker

```powershell
# Testler
pytest tests/ -v

# Docker
docker build -t greenedge-5g:latest .
docker run -p 8000:8000 greenedge-5g:latest
```

---

## Sistem Mimarisi

| Katman | Teknoloji | İşlev |
|--------|-----------|-------|
| **UI** | Streamlit | KPI kartları, grafikler |
| **API** | FastAPI | POST /decision, GET /health |
| **RL Motor** | Stable-Baselines3, PyTorch | PPO karar üretimi |
| **Simülatör** | Gymnasium | MDP ortamı, ödül hesaplama |

---

## Problem Tanımı

5G ağlarında sabit kurallı yönlendirme (en hızlı sunucu, CPU eşiği vb.) zamansal etkileri göz ardı eder. Art arda aynı sunucuya yönlendirme, SLA ihlallerine yol açabilir.

GreenEdge-5G bu problemi MDP olarak modelleyerek dinamik optimizasyon sağlar.

---

## Teknik Detaylar

### Durum Vektörü (Observation Space)

| İndeks | Değişken | Açıklama | Aralık |
|--------|----------|----------|--------|
| 0 | `cpu_a` | Edge-A CPU yükü | [0, 1] |
| 1 | `cpu_b` | Edge-B CPU yükü | [0, 1] |
| 2 | `q_a` | Edge-A kuyruk oranı | [0, 1] |
| 3 | `q_b` | Edge-B kuyruk oranı | [0, 1] |
| 4 | `link_q` | Bağlantı kalitesi | [0, 1] |
| 5 | `energy_price` | Enerji fiyatı | [0, 1] |

### Eylem Uzayı (Action Space)

| Eylem | Hedef | Açıklama |
|-------|-------|----------|
| 0 | Edge-A | Düşük gecikme, yüksek enerji |
| 1 | Edge-B | Orta gecikme, orta enerji |
| 2 | Cloud | Yüksek gecikme, düşük enerji |

### Ödül Fonksiyonu

```
Reward = – ( α × Energy_norm + β × Latency_norm + γ × SLA_penalty )
```

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| α (alpha) | 0.3 | Enerji ağırlığı |
| β (beta) | 0.5 | Gecikme ağırlığı |
| γ (gamma) | 0.2 | SLA ceza ağırlığı |

### Politika Karşılaştırması

| Politika | Etiket | Açıklama |
|----------|--------|----------|
| **Hız** | greedy_min_latency | En düşük gecikmeli sunucu |
| **Maliyet** | greedy_min_energy | En düşük enerjili sunucu |
| **Yük** | simple_threshold | CPU eşik tabanlı yönlendirme |
| **PPO** | rl_ppo | Yapay zeka ile dinamik optimizasyon |

### Güven Skoru ve Fallback

Her karar için güven skoru üretilir: `Güven = max_prob - second_max_prob`

| Durum | Aksiyon |
|-------|---------|
| Güven ≥ threshold | PPO kararı |
| Güven < threshold | Fallback politikası (Yük) |

---

## Dashboard Özellikleri

- **Kontrol Paneli**: Politika seçimi (Hız/Maliyet/Yük/PPO)
- **KPI Kartları**: Ortalama gecikme, p95 gecikme, Enerji/Mbps, SLA ihlal oranı
- **Grafikler**: Ödül trendi, gecikme dağılımı, enerji tüketimi, politika karşılaştırma

---

## Sonuçlar (Simülasyon)

200 episode, ~10.000 karar adımı üzerinde:

| Metrik | İyileşme |
|--------|----------|
| Ortalama enerji tüketimi | ≈ %18 azalma |
| p95 gecikme | ≈ %12 azalma |
| SLA ihlal oranı | ≈ %40 azalma |

*Klasik "en düşük gecikme" ve eşik tabanlı politikalara kıyasla*

---

## Demo Senaryosu

```powershell
# 1. Dashboard başlat
streamlit run greenedge/dashboard/app.py

# 2. Politika karşılaştır: Önce "Hız", sonra "PPO" seç
# 3. KPI farklarını gözlemle
```

---

## Proje Yapısı

```
greenedge-mvp/
├── greenedge/
│   ├── simulator/         # Gymnasium ortamı, ödül hesaplama
│   ├── rl/                # PPO eğitim, değerlendirme, baseline'lar
│   ├── api/               # FastAPI endpoints
│   └── dashboard/         # Streamlit UI
├── tests/                 # Pytest testleri
├── experiments/           # Sonuçlar, model dosyaları
├── k8s/                   # Kubernetes deployment
├── Dockerfile             # Container build
├── requirements.txt       # Production bağımlılıkları
└── requirements-dev.txt   # Geliştirme bağımlılıkları
```

---

## CI/CD

GitHub Actions ile otomatik pipeline: Ruff lint, Pytest, Mypy type check, Docker build.

---

*GreenEdge-5G - Akıllı 5G Trafik Yönetimi*
