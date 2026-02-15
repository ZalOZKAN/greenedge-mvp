# GreenEdge-5G

**5G Kenar/Bulut Altyapısı için Yapay Zeka Destekli İş Yükü Yönlendirme Sistemi**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Proje Özeti

GreenEdge-5G, 5G ağlarında iş yüklerini **kenar sunucuları (edge-a, edge-b)** ve **bulut** arasında akıllıca yönlendiren bir yapay zeka sistemidir. Pekiştirmeli öğrenme (Reinforcement Learning) kullanarak:

- ⚡ **Gecikmeyi minimize eder** (ortalama <100 ms hedefi)
- 🔋 **Enerji tüketimini azaltır**
- 📊 **SLA ihlallerini sıfıra yakın tutar** (<%5 hedefi)

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                      GreenEdge-5G                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │   Edge-A    │    │   Edge-B    │    │    Cloud    │    │
│   │  (Kenar-A)  │    │  (Kenar-B)  │    │   (Bulut)   │    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│          │                  │                  │            │
│          └──────────────────┼──────────────────┘            │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │   YZ Karar      │                      │
│                    │   Motoru (PPO)  │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│          ┌──────────────────┼──────────────────┐            │
│          │                  │                  │            │
│   ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐    │
│   │  Simülatör  │    │   FastAPI   │    │  Dashboard  │    │
│   │  (Gym Env)  │    │     API     │    │ (Streamlit) │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Proje Yapısı

```
greenedge-mvp/
├── greenedge/
│   ├── simulator/          # Gymnasium ortamı
│   │   ├── env.py          # GreenEdgeEnv sınıfı
│   │   ├── config.py       # Ortam konfigürasyonu
│   │   └── smoke_test.py   # Hızlı test
│   ├── rl/
│   │   ├── train.py        # PPO/DQN eğitimi
│   │   ├── evaluate.py     # Model değerlendirme
│   │   └── baselines.py    # Karşılaştırma politikaları
│   ├── api/
│   │   └── main.py         # FastAPI endpoint'leri
│   └── dashboard/
│       └── app.py          # Streamlit arayüzü
├── experiments/
│   ├── policy.zip          # Eğitilmiş PPO modeli
│   ├── results.json        # Değerlerlendirme sonuçları
│   ├── plots_reward.png    # Ödül grafiği
│   └── plots_tradeoff.png  # Trade-off grafiği
├── docs/
│   ├── demo_steps.md       # Demo adımları
│   └── mvp_report.md       # MVP raporu
├── k8s/                    # Kubernetes deployment
├── requirements.txt        # Bağımlılıklar
├── AGENTS.md              # Agent talimatları
└── README.md              # Bu dosya
```

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/your-repo/greenedge-mvp.git
cd greenedge-mvp

# Virtual environment oluştur
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Smoke Test (Doğrulama)

```bash
python -m greenedge.simulator.smoke_test
```

Beklenen çıktı:
```
Sample Observation: [0.45, 0.32, 5, 3, 0.8, 0.15]
Sample Actions: edge-a, edge-b, cloud
Latency: 85.2 ms, Energy: 0.0023
```

### 3. Model Eğitimi

```bash
# PPO ile 20,000 adım eğitim
python -m greenedge.rl.train --algo ppo --steps 20000

# Veya DQN ile
python -m greenedge.rl.train --algo dqn --steps 20000
```

### 4. Model Değerlendirme

```bash
python -m greenedge.rl.evaluate --episodes 200
```

Çıktılar:
- `experiments/results.json` - Metrikler
- `experiments/plots_reward.png` - Ödül grafiği
- `experiments/plots_tradeoff.png` - Trade-off grafiği

### 5. Dashboard Başlatma

```bash
streamlit run greenedge/dashboard/app.py
```

Tarayıcıda aç: http://localhost:8501

### 6. API Başlatma

```bash
python -m greenedge.api.main
```

API: http://localhost:8000/docs

---

## 📊 Performans Sonuçları

| Politika | Ödül | Gecikme (ms) | P95 (ms) | Enerji | SLA İhlal |
|----------|------|--------------|----------|--------|-----------|
| **YZ (PPO)** | **-17.41** | **95.2** | **108.0** | 0.7215 | **1.8%** |
| Hız | -18.06 | 98.4 | 121.6 | 0.7280 | 5.8% |
| Maliyet | -20.61 | 111.4 | 178.0 | **0.7023** | 24.0% |
| CPU | -18.60 | 100.6 | 130.4 | 0.7085 | 12.6% |

**🏆 Kazanan: YZ (PPO)** - En yüksek ödül, en düşük gecikme, en az SLA ihlali

---

## 🎯 Observation & Action Space

### Observation (Gözlem Vektörü)
| Index | Değişken | Açıklama | Aralık |
|-------|----------|----------|--------|
| 0 | cpu_a | Edge-A CPU kullanımı | [0, 1] |
| 1 | cpu_b | Edge-B CPU kullanımı | [0, 1] |
| 2 | q_a | Edge-A kuyruk uzunluğu | [0, 20] |
| 3 | q_b | Edge-B kuyruk uzunluğu | [0, 20] |
| 4 | link_q | Bağlantı kalitesi | [0, 1] |
| 5 | energy_price | Anlık enerji fiyatı | [0, 1] |

### Action (Eylem)
| Değer | Hedef | Açıklama |
|-------|-------|----------|
| 0 | edge-a | Kenar sunucu A'ya yönlendir |
| 1 | edge-b | Kenar sunucu B'ye yönlendir |
| 2 | cloud | Bulut'a yönlendir |

### Reward (Ödül Fonksiyonu)
```
reward = -(α × energy + β × latency + γ × sla_penalty)
```
- α = 0.3 (enerji ağırlığı)
- β = 0.5 (gecikme ağırlığı)
- γ = 10.0 (SLA ceza ağırlığı)

---

## 🖥️ Dashboard Özellikleri

1. **Değerlendirme Özeti** - KPI kartları ve terim açıklamaları
2. **Politika Karşılaştırması** - Tüm politikaların tablo görünümü
3. **Trade-off Grafiği** - Gecikme vs Enerji scatter plot
4. **Canlı Simülasyon** - Seçilen politika ile 50 adımlık demo
5. **A/B Test Paneli** - İki politikayı yan yana karşılaştır
6. **PDF Rapor** - Tek tıkla indirilebilir rapor

---

## 🔌 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/health` | GET | Sağlık kontrolü |
| `/decision` | POST | Karar al (obs → action) |
| `/docs` | GET | Swagger UI |

### Örnek İstek
```bash
curl -X POST http://localhost:8000/decision \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.5, 0.3, 2, 1, 0.9, 0.2]}'
```

### Örnek Yanıt
```json
{
  "action": 0,
  "target": "edge-a",
  "confidence": 0.85,
  "predicted_latency": 82.5,
  "predicted_energy": 0.0021
}
```

---

## 🧪 Testler

```bash
# Simulator smoke test
python -m greenedge.simulator.smoke_test

# Model evaluation
python -m greenedge.rl.evaluate --episodes 50
```

---

## 📦 Bağımlılıklar

- **Python** >= 3.10
- **gymnasium** >= 0.29.0
- **stable-baselines3** >= 2.0.0
- **torch** >= 2.0.0
- **fastapi** >= 0.100.0
- **streamlit** >= 1.25.0
- **plotly** >= 5.15.0
- **reportlab** >= 4.0.0

Tam liste: [requirements.txt](requirements.txt)

---

## 🤝 Katkıda Bulunma

1. Fork et
2. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
3. Commit et (`git commit -m 'Add amazing feature'`)
4. Push et (`git push origin feature/amazing-feature`)
5. Pull Request aç

---

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👥 Ekip

- **Proje Sahibi**: [İsim]
- **Geliştirici**: [İsim]

---

## 📞 İletişim

Sorularınız için: [email@example.com]

---

<p align="center">
  <b>GreenEdge-5G</b> - Akıllı 5G İş Yükü Yönlendirme
</p>
