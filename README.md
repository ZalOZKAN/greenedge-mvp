# GreenEdge-5G

5G şebekelerinde enerji ve gecikmeyi birlikte optimize eden,
Cloud-Native mimariye sahip yapay zekâ tabanlı akıllı trafik yönetim sistemi.

[![CI](https://github.com/your-org/greenedge-mvp/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/greenedge-mvp/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. (Opsiyonel) Geliştirme bağımlılıklarını yükle
pip install -r requirements-dev.txt
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

### Testleri Çalıştırma

```powershell
# Tüm testleri çalıştır
pytest tests/ -v

# Coverage ile çalıştır
pytest tests/ --cov=greenedge --cov-report=html

# Sadece belirli bir test dosyası
pytest tests/test_api.py -v
```

### Docker ile Çalıştırma

```powershell
# Docker image oluştur
docker build -t greenedge-5g:latest .

# Container başlat
docker run -p 8000:8000 greenedge-5g:latest

# Environment variables ile
docker run -p 8000:8000 \
  -e GREENEDGE_LOG_LEVEL=DEBUG \
  -e GREENEDGE_API_KEY=your-secret-key \
  greenedge-5g:latest
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
| GET | `/docs` | Swagger UI dokümantasyonu |
| GET | `/openapi.json` | OpenAPI şeması |

### Güvenlik

API, opsiyonel güvenlik özellikleri sunar:

**API Key Authentication:**
```powershell
# .env dosyasında API key tanımla
GREENEDGE_API_KEY=your-secret-key

# İstek gönderirken header ekle
curl -H "X-API-Key: your-secret-key" http://localhost:8000/decision
```

**Rate Limiting:**
- Varsayılan: 60 istek/dakika/IP
- Response header'larında kalan limit gösterilir:
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Limit`

### POST /decision

**Request Body:**
```json
{
  "obs": [0.3, 0.5, 0.2, 0.25, 0.8, 0.4]
}
```

**Response:**
```json
{
  "action": 1,
  "action_label": "edge-b",
  "confidence": 0.85,
  "fallback_used": false,
  "predicted_kpis": {
    "latency_ms": 45.2,
    "energy_per_mbps": 0.62,
    "sla_violation": 0
  }
}
```

---

## Konfigürasyon

Sistem ayarları environment variables ile yönetilir. `.env.example` dosyasını kopyalayarak başlayın:

```powershell
copy .env.example .env
```

### Environment Variables

| Değişken | Varsayılan | Açıklama |
|----------|------------|----------|
| `GREENEDGE_HOST` | 127.0.0.1 | API sunucu host |
| `GREENEDGE_PORT` | 8000 | API sunucu port |
| `GREENEDGE_LOG_LEVEL` | INFO | Log seviyesi (DEBUG/INFO/WARNING/ERROR) |
| `GREENEDGE_CONFIDENCE_THRESHOLD` | 0.55 | Güven eşiği |
| `GREENEDGE_SLA_MS` | 120.0 | SLA gecikme limiti (ms) |
| `GREENEDGE_API_KEY` | (boş) | API key (boş = auth kapalı) |
| `GREENEDGE_RATE_LIMIT_PER_MINUTE` | 60 | Rate limit (istek/dakika) |

### Ödül Ağırlıkları

Ödül fonksiyonu parametreleri `greenedge/simulator/config.py` dosyasında tanımlıdır:

```python
@dataclass
class RewardWeights:
    alpha: float = 0.35   # Enerji ağırlığı
    beta: float = 0.55    # Gecikme ağırlığı
    gamma: float = 0.10   # SLA ceza ağırlığı
```

---

## Proje Yapısı

```
greenedge-mvp/
├── greenedge/
│   ├── __init__.py
│   ├── logging_config.py  # Merkezi logging yapılandırması
│   ├── settings.py        # Environment variables yönetimi
│   ├── py.typed           # PEP 561 type hint marker
│   ├── simulator/         # Gymnasium ortamı + ödül + senaryo üretimi
│   │   ├── env.py         # Ana simülasyon ortamı
│   │   ├── config.py      # Simülasyon parametreleri
│   │   └── smoke_test.py
│   ├── rl/                # Eğitim + değerlendirme + baseline'lar
│   │   ├── train.py       # PPO eğitim scripti
│   │   ├── evaluate.py    # Değerlendirme + metrikler
│   │   └── baselines.py   # Kural tabanlı politikalar
│   ├── api/               # FastAPI endpoints
│   │   ├── main.py        # Ana API uygulaması
│   │   └── security.py    # Rate limiting + API key auth
│   └── dashboard/         # Streamlit UI
│       ├── app.py
│       └── style.css
├── tests/                 # Pytest test dosyaları
│   ├── test_env.py        # Simülatör testleri
│   ├── test_baselines.py  # Baseline politika testleri
│   └── test_api.py        # API endpoint testleri
├── experiments/           # Sonuçlar, plotlar, model dosyaları
│   └── results.json
├── docs/                  # Dokümantasyon
│   ├── demo_steps.md
│   ├── mvp_report.md
│   └── AGENTS.md
├── k8s/                   # Kubernetes deployment dosyaları
├── .github/workflows/     # CI/CD pipeline
│   └── ci.yml
├── Dockerfile             # Container build
├── .dockerignore
├── .env.example           # Örnek environment config
├── pytest.ini             # Test yapılandırması
├── ruff.toml              # Linter yapılandırması
├── requirements.txt       # Production bağımlılıkları
└── requirements-dev.txt   # Geliştirme bağımlılıkları
```

---

## Kubernetes Deployment

Proje, k3s veya herhangi bir Kubernetes cluster üzerinde container olarak çalıştırılabilir.

### Docker Image Build

```bash
# Image oluştur
docker build -t greenedge-5g:latest .

# (Opsiyonel) Registry'ye push et
docker tag greenedge-5g:latest your-registry/greenedge-5g:latest
docker push your-registry/greenedge-5g:latest
```

### Kubernetes Deployment

```bash
# Deployment uygula
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/app-service.yaml

# Podları kontrol et
kubectl get pods

# Logları izle
kubectl logs -f deployment/edge-echo
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
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Port çakışması | `--port 8502` flag'i ekle |
| `401 Unauthorized` | `X-API-Key` header'ını kontrol et |
| `429 Too Many Requests` | Rate limit - 60 saniye bekle |
| Docker build hatası | `docker system prune` ile cache temizle |
| Test hatası | `pip install -r requirements-dev.txt` |

---

## Bağımlılıklar

### Production Dependencies

```
numpy==1.26.4
gymnasium==0.29.1
stable-baselines3==2.3.2
torch==2.2.2
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
streamlit==1.36.0
plotly==5.22.0
pandas==2.2.2
matplotlib==3.9.0
reportlab==4.2.0
python-dotenv==1.0.1
```

### Development Dependencies

```
pytest==8.2.2
pytest-cov==5.0.0
httpx==0.27.0
ruff==0.4.10
mypy==1.10.0
pre-commit==3.7.1
```

---

## Gelecek Aşamalar

- [x] Unit test altyapısı (pytest)
- [x] Docker containerization
- [x] CI/CD pipeline (GitHub Actions)
- [x] API güvenlik (rate limiting, API key)
- [x] Centralized logging
- [x] Environment-based configuration
- [ ] Geniş ölçekli ağ simülasyon doğrulaması
- [ ] Emülasyon testleri
- [ ] Operatör verisi ile kontrollü pilot
- [ ] SDN entegrasyonu
- [ ] Model versiyonlama (MLflow)
- [ ] Dashboard modülerleştirme

---

## CI/CD

GitHub Actions ile otomatik CI pipeline:

- **Lint**: Ruff ile kod kalitesi kontrolü
- **Test**: Pytest ile unit testler
- **Type Check**: Mypy ile tip kontrolü
- **Docker Build**: Container image oluşturma

Her push ve pull request'te otomatik çalışır.

---

## Lisans

Bu proje MIT lisansı altında sunulmaktadır.

---

*GreenEdge-5G - Akıllı 5G Trafik Yönetimi*
