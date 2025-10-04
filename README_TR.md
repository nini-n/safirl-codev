# SafeRL for Franka (MuJoCo) — CBF/MPC Shielded PPO

Planar **Franka** benzeri 3-DoF kol için **güvenli pekiştirmeli öğrenme** iskeleti: PPO tabanlı politika + **CBF** ve **MPC** güvenlik kalkanları (shield). Eğitim, değerlendirme, benchmark, grafik üretimi ve rapor görselleri tek adımda veya ayrı ayrı çalıştırılabilir. Tüm deneyler **sanal ortamda** (MuJoCo veya kinematik) çalışır.

> ✅ **Durum:** v1.0.0 — Eğitim/değerlirme/benchmark betikleri, grafik üretimi ve rapor toplayıcı hazır
> ✅ **Testler:** `pytest` ile 4 test geçiyor
> ✅ **Docker:** Başsız (headless) kullanım için hazır imaj

> [English](README.md) | Türkçe
---

## İçindekiler
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Eğitim](#eğitim)
- [Değerlendirme](#değerlendirme)
- [Benchmark](#benchmark)
- [Grafikler ve Rapor](#grafikler-ve-rapor)
- [Testler](#testler)
- [Deneyleri Uçtan Uca Tek Komutla Yeniden Üretme](#deneyleri-uçtan-uca-tek-komutla-yeniden-üretme)
- [Docker ile Çalıştırma](#docker-ile-çalıştırma)
- [Proje Yapısı](#proje-yapısı)
- [Atıf](#atıf)
- [Lisans](#lisans)
- [Sorun Giderme (FAQ)](#sorun-giderme-faq)

---

## Kurulum

> Windows (CMD) örnekleri verilmiştir. Python **3.9** önerilir.

1) Depoyu klonla ve dizine geç:
```bat
git clone https://github.com/<kullanici>/<repo-adi>.git
cd <repo-adi>
```

2) (Önerilir) Sanal ortam:
```bat
python -m venv .venv
.venv\Scripts\activate
```

3) Bağımlılıklar:
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest ruff
```

> Not: `torchaudio/torchvision` **zorunlu değildir**. Proje **torch==2.2.2** ile test edilmiştir.

---

## Hızlı Başlangıç

Eğitimi atlayıp mevcut politikayla (repo içindeki `runs/latest.pt`) **hızlı üretim**:
```bat
python scripts\reproduce_all.py --skip-train
```
Tüm çıktı ve görseller `runs/` klasörüne, rapor görselleri `runs/report/` altına kaydedilir.

---

## Eğitim

Temel ayarlar `experiments/base.yaml` içindedir. Örnek eğitim (20000 adım):
```bat
python scripts\train.py --cfg experiments\base.yaml --steps 20000
```
Eğitim sonunda politika `runs/latest.pt` olarak kaydedilir ve eğitim metrikleri `runs/` altına yazılır.

> Eğer `ppo.py` için minimal arayüzü kullanıyorsanız `train.py` dosyasında `agent.act()` yerine `agent.mu(o)` + örnekleme kullanımı yapılır.

---

## Değerlendirme

Mevcut politikayla kısa değerlendirme (5 bölüm):
```bat
python scripts\evaluate.py --cfg experiments\base.yaml --policy runs\latest.pt --episodes 5
```
> Çıktıda bölüm başına toplam ödül, ihlal bilgisi ve STL-robustness özetleri yazdırılır.

---

## Benchmark

Korumasız (**no_shield**), **CBF** ve **MPC** kalkanlarıyla kıyas:
```bat
python scripts\benchmark.py --cfg experiments\base.yaml --policy runs\latest.pt --episodes 8
```
Özet `.csv` dosyaları:
- Bölüm-bazında: `runs/benchmark.csv`
- Toplam özet: `runs/benchmark_summary.csv`

Grafikleri üretmek için:
```bat
python scripts\plot_benchmark.py
```

---

## Grafikler ve Rapor

Eğitim/değerlendirme eğrileri:
```bat
python scripts\plot_metrics.py
```
Rapor dosyalarını tek klasörde topla:
```bat
python scripts\make_report_assets.py
```
> Görseller `runs/` ve `runs/report/` altına kaydedilir.

---

## Testler

Hızlı doğrulama:
```bat
pytest -q
```
> Şu an 4 test **geçiyor**.

---

## Deneyleri Uçtan Uca Tek Komutla Yeniden Üretme

Eğitimi de içeren uçtan uca tekrar üretim:
```bat
python scripts\reproduce_all.py --steps 20000
```
Sadece mevcut politikayla hızlı üretim:
```bat
python scripts\reproduce_all.py --skip-train
```

---

## Docker ile Çalıştırma

> Windows’ta Docker Desktop kurulu ve çalışır olmalı.

İmajı oluştur:
```bat
docker build -t safirl .
```

Yerel `runs/` klasörünü container’a bağlayıp raporları üret:
```bat
docker run --rm -it -v %CD%\runs:/app/runs safirl
```
> Container içinde betikler sırasıyla `evaluate → benchmark → plot` çalıştırır ve çıktılar host’taki `runs/` klasörüne düşer.

---

## Proje Yapısı

```
.
├─ assets/
│  └─ franka/franka.xml
├─ envs/
│  ├─ __init__.py
│  ├─ franka_kinematic_env.py
│  └─ franka_mujoco_env.py
├─ rl/
│  ├─ __init__.py
│  └─ ppo.py
├─ shield/
│  ├─ __init__.py
│  ├─ cbf_qp.py
│  └─ mpc_shield.py
├─ specs/
│  ├─ __init__.py
│  └─ specs.py
├─ verify/
│  ├─ __init__.py
│  └─ robustness.py
├─ scripts/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ benchmark.py
│  ├─ plot_metrics.py
│  ├─ plot_benchmark.py
│  ├─ make_report_assets.py
│  ├─ apply_best_safety.py
│  └─ codesign.py
├─ tests/
│  ├─ test_kinematics.py
│  ├─ test_specs.py
│  └─ test_shields.py
├─ runs/                  (çıktı; git’e dahil ETME)
├─ requirements.txt
├─ Dockerfile
├─ README.md
└─ .github/workflows/ci.yml

```

---

## Atıf

Bu depoyu akademik çalışmalarda kullanırsanız aşağıdaki bibtex benzeri atıfı ekleyebilirsiniz:

```
@software{SafirlCodev2025,
  author  = {Nihan <soyad>},
  title   = {SafeRL for Franka (MuJoCo) — Shielded PPO with CBF/MPC},
  year    = {2025},
  url     = {https://github.com/<kullanici>/<repo-adi>}
}
```

---

## Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Ayrıntılar için `LICENSE` dosyasına bakınız.

---

## Sorun Giderme (FAQ)

**Docker build sırasında `libgl1-mesa-glx` bulunamadı**
Debian *trixie* tabanlı imajlarda paket adı değişiklik gösterebilir. Bu repo Dockerfile’ında GL/X bağımlılıkları güncellenmiştir. Yine de sorun yaşarsanız şu minimal set işinizi görür: `libgl1 libxrender1 libxext6 libxi6 libxxf86vm1 libxrandr2 libxcb1`.

**`PPOAgent` ile `act` hatası**
Bu repoda minimal arayüz kullanılıyor: `mu(obs)` deterministik ortalama eylem verir, eğitimde örnekleme `torch.distributions.Normal(mu, std)` üzerinden yapılır. Scriptler buna göre günceldir.

**MuJoCo XML bulunamadı**
`assets/franka/franka.xml` yolunu denetleyin. OneDrive masaüstü yol sorunları için kodda **path fallback** bulunmaktadır; yine de dosyanın var olduğundan emin olun.

**Torchaudio/Torchvision uyumsuzluk uyarıları**
Bu paketler zorunlu değil; `torch==2.2.2` ile proje test edilmiştir. Uyarılar göz ardı edilebilir.


