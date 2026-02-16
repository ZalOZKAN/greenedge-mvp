# GreenEdge-5G

5G şebekelerinde enerji ve gecikmeyi birlikte optimize eden,
Cloud-Native mimariye sahip yapay zekâ tabanlı akıllı trafik yönetim sistemi.

---

## Hızlı Bakış (Quick Look)

GreenEdge-5G, modern 5G altyapılarında iş yükü yönlendirme kararlarını
yapay zekâ ile veren, Kubernetes (k3s) üzerinde çalışabilen ve
REST API aracılığıyla entegre edilebilen bir karar motorudur.

Proje üç temel gücü birleştirir:

- Deep Reinforcement Learning (PPO algoritması)
- Cloud-Native mimari (k3s / container deployment)
- Gerçek zamanlı KPI görselleştirme ve raporlama

Sistem yalnızca bir simülasyon değildir; modüler yapısı sayesinde
operatör altyapılarına entegre edilebilir bir karar katmanı olarak tasarlanmıştır.

---

## 5 Dakikada Demo Çalıştırma

### Kurulum

```powershell
# 1. Repository'yi klonla
git clone https://github.com/your-org/greenedge-mvp.git
cd greenedge-mvp

# 2. Virtual environment oluştur
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### Çalıştırma Komutları

```powershell
# Simülatör smoke test
python -m greenedge.simulator.smoke_test

# Model eğitimi (20.000 adım)
python -m greenedge.rl.train --algo ppo --steps 20000

# Model değerlendirmesi (200 episode)
python -m greenedge.rl.evaluate --episodes 200

# API sunucusu başlat
python -m greenedge.api.main

# Dashboard başlat
streamlit run greenedge/dashboard/app.py
```

---

## Sistem Mimarisi

Mimari dört ana katmandan oluşur:

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard (Streamlit)                │
│        KPI Kartları • Grafikler • PDF Rapor            │
├─────────────────────────────────────────────────────────┤
│                    REST API (FastAPI)                   │
│            POST /decision • GET /health                 │
├─────────────────────────────────────────────────────────┤
│                  RL Karar Motoru (PPO)                  │
│         Stable-Baselines3 • PyTorch • Güven Skoru      │
├─────────────────────────────────────────────────────────┤
│                Simülatör (Gymnasium Env)                │
│            Durum Vektörü • Ödül Fonksiyonu             │
└─────────────────────────────────────────────────────────┘
```

---

## Problem Tanımı

5G ağlarında gelen kullanıcı talepleri genellikle sabit kurallarla yönlendirilir:

- En hızlı sunucuyu seç
- En ucuz kaynağı seç
- CPU eşiğine göre yönlendir

Bu yöntemler zamansal etkiyi dikkate almaz.
Aynı edge sunucusunun art arda seçilmesi,
ilerleyen adımlarda gecikme artışına ve SLA ihlaline yol açabilir.

GreenEdge-5G, bu problemi Markov Karar Süreci (MDP) olarak modelleyerek
dinamik ve öğrenebilen bir çözüm sunar.

---

## Teknik Detaylar

### Durum Vektörü (Observation Space)

Sistemin anlık durumunu tanımlayan 6 boyutlu vektör:

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

---

### Ödül Fonksiyonu

```
Reward = – ( α × Energy_norm + β × Latency_norm + γ × SLA_penalty )
```

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| α (alpha) | 0.3 | Enerji ağırlığı |
| β (beta) | 0.5 | Gecikme ağırlığı |
| γ (gamma) | 0.2 | SLA ceza ağırlığı |

Bu katsayılar `config.yaml` üzerinden ayarlanabilir.

---

### Politika Karşılaştırması

Sistem dört farklı politika sunar:

| Politika | Etiket | Açıklama |
|----------|--------|----------|
| **Hız** | greedy_min_latency | Her zaman en düşük gecikmeli sunucuyu seç |
| **Maliyet** | greedy_min_energy | Her zaman en düşük enerjili sunucuyu seç |
| **Yük** | simple_threshold | CPU yüküne göre eşik tabanlı yönlendirme |
| **PPO** | rl_ppo | Yapay zeka ile dinamik optimizasyon |

---

### Algoritma: PPO (Proximal Policy Optimization)

Seçilme nedenleri:

- Kararlı yakınsama
- Hiperparametre hassasiyetinin düşük olması
- Sürekli durum uzaylarında iyi performans
- Endüstride yaygın kullanım

Eğitim altyapısı:

- Gymnasium (özel ortam)
- Stable-Baselines3
- PyTorch

---

### Güven Skoru ve Fallback Mekanizması

Her karar için bir güven skoru (0-1 arası) üretilir:

```
Güven = max_probability - second_max_probability
```

| Durum | Aksiyon |
|-------|---------|
| Güven ≥ threshold | PPO kararı kullanılır |
| Güven < threshold | Fallback politikasına geç (varsayılan: Yük) |

Bu yapı, endüstriyel entegrasyon için güvenli bir mimari sağlar.

---

## Dashboard Özellikleri

Dashboard üzerinde görüntülenen bileşenler:

### Kontrol Paneli
- Politika seçimi: Hız / Maliyet / Yük / PPO
- Simülasyon parametreleri

### KPI Kartları
- **Ortalama Gecikme (ms)**: Tüm kararların ortalama gecikme süresi
- **p95 Gecikme (ms)**: %95'lik gecikme dilimi
- **Enerji/Mbps**: Birim veri başına enerji tüketimi
- **SLA İhlali (%)**: Gecikme eşiğini aşan karar oranı

### Grafikler
- Ödül trendi (episode bazlı)
- Gecikme dağılımı (p50/p95)
- Enerji tüketim trendi
- Politika karşılaştırma çubuğu

### PDF Rapor İndirme
"Raporu indir (PDF)" butonu ile tek tıkla rapor oluşturma:
- Seçili politika
- Metrik tablosu
- Grafikler
- Zaman damgası
- Git commit hash
- Konfigürasyon özeti

---

## MVP Sonuçları (Simülasyon Tabanlı Ön Testler)

Testler kontrollü simülasyon ortamında gerçekleştirilmiştir.
Gerçek saha verisi kullanılmamıştır.

200 bölüm ve yaklaşık 10.000 karar adımı üzerinden yapılan değerlendirmede:

| Metrik | İyileşme |
|--------|----------|
| Ortalama enerji tüketimi | ≈ %18 azalma |
| p95 gecikme | ≈ %12 azalma |
| SLA ihlal oranı | ≈ %40 azalma |

Bu sonuçlar, klasik "en düşük gecikme" ve eşik tabanlı politikalara kıyasla elde edilmiştir.

---

## API Referansı

### Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Sistem sağlık kontrolü |
| POST | `/decision` | Karar al |

### POST /decision

**Request Body:**
```json
{
  "observation": [0.7, 0.3, 0.5, 0.2, 0.8, 0.4]
}
```

**Response:**
```json
{
  "action": 1,
  "action_name": "edge-b",
  "confidence": 0.85,
  "predicted_kpis": {
    "latency_ms": 12.5,
    "energy_wh": 0.8
  }
}
```

---

## Konfigürasyon

Sistem ayarları `config.yaml` dosyasından yönetilir:

```yaml
# Ödül ağırlıkları
reward:
  alpha: 0.3      # Enerji ağırlığı
  beta: 0.5       # Gecikme ağırlığı
  gamma: 0.2      # SLA ceza ağırlığı

# Güven mekanizması
confidence:
  threshold: 0.6
  fallback_policy: "simple_threshold"

# Eğitim parametreleri
training:
  default_steps: 20000
  evaluation_episodes: 200

# Varsayılan politika
default_policy: "PPO"
```

---

## Proje Yapısı

```
greenedge-mvp/
├── greenedge/
│   ├── simulator/       # Gymnasium ortamı + ödül + senaryo üretimi
│   │   ├── env.py       # Ana simülasyon ortamı
│   │   ├── config.py    # Simülasyon parametreleri
│   │   └── smoke_test.py
│   ├── rl/              # Eğitim + değerlendirme + baseline'lar
│   │   ├── train.py     # PPO eğitim scripti
│   │   ├── evaluate.py  # Değerlendirme + metrikler
│   │   └── baselines.py # Kural tabanlı politikalar
│   ├── api/             # FastAPI endpoints
│   │   └── main.py
│   └── dashboard/       # Streamlit UI
│       ├── app.py
│       └── style.css
├── experiments/         # Sonuçlar, plotlar, log dosyaları
│   └── results.json
├── docs/                # Dokümantasyon
│   ├── demo_steps.md
│   └── mvp_report.md
├── k8s/                 # Kubernetes deployment dosyaları
├── config.yaml          # Merkezi konfigürasyon
└── requirements.txt
```

---

## Kubernetes Deployment

Proje, k3s üzerinde container olarak çalıştırılabilir.

Kullanılan dosyalar:
- `k8s/app-deployment.yaml`
- `k8s/app-service.yaml`

```bash
# Deployment uygula
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/app-service.yaml

# Podları kontrol et
kubectl get pods
```

---

## Demo Senaryosu (Jüri için 60-120 sn)

### 1. Dashboard'u Aç
```powershell
streamlit run greenedge/dashboard/app.py
```

### 2. Politika Seç ve Karşılaştır
- Önce "Hız" politikasını seç → KPI'ları gözlemle
- Sonra "PPO" politikasını seç → Farkı göster

### 3. Sonuçları Açıkla
- "PPO politikası, Hız politikasına göre %18 daha az enerji kullanırken gecikmeyi %12 düşürüyor"

### 4. Raporu İndir
- "Raporu indir (PDF)" butonuna tıkla
- PDF dosyasını aç ve göster

---

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `results.json bulunamadı` | `python -m greenedge.rl.evaluate` çalıştır |
| `Model dosyası yok` | `python -m greenedge.rl.train` çalıştır |
| `Module not found` | `pip install -r requirements.txt` |
| Port çakışması | `--port 8502` flag'i ekle |

---

## Bağımlılıklar

Minimal bağımlılık listesi:

```
numpy
gymnasium
stable-baselines3
torch
fastapi
uvicorn
pydantic
streamlit
matplotlib
plotly
```

---

## Gelecek Aşamalar

- Geniş ölçekli ağ simülasyon doğrulaması
- Emülasyon testleri
- Operatör verisi ile kontrollü pilot
- SDN entegrasyonu
- Model versiyonlama ve CI/CD

---

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

*GreenEdge-5G - Akıllı 5G Trafik Yönetimi*
