# DPM-Solver: Solver ODE Cepat untuk Sampling Diffusion Probabilistic Model dalam Sekitar 10 Langkah

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LuChengTHU/dpmsolver_sdm)

Kode resmi untuk paper [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) (**Neurips 2022 Oral**) dan [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) oleh [Cheng Lu](https://luchengthu.github.io/), [Yuhao Zhou](https://yuhaoz.com/), [Fan Bao](https://baofff.github.io/), [Jianfei Chen](https://ml.cs.tsinghua.edu.cn/~jianfei/), [Chongxuan Li](https://zhenxuan00.github.io/) dan [Jun Zhu](https://ml.cs.tsinghua.edu.cn/~jun/index.shtml).

--------------------

## 📖 Pengenalan DPM-Solver++ 2M

DPM-Solver++ adalah versi perbaikan dari DPM-Solver yang dirancang untuk mempercepat sampling pada diffusion probabilistic models. **DPM-Solver++ 2M** mengacu pada konfigurasi khusus:

- **DPM-Solver++**: Algoritma yang menggunakan data prediction model (memprediksi `x_0` / data asli) daripada noise prediction model
- **2**: Order kedua (second-order) - menggunakan informasi dari 2 langkah sebelumnya
- **M**: Multistep - metode yang menggunakan informasi dari beberapa langkah sebelumnya (seperti metode Adams-Bashforth)

DPM-Solver++ 2M sangat cocok untuk **guided sampling** (sampling dengan conditioning) terutama dengan guidance scale yang besar, dan dapat menghasilkan sampel berkualitas tinggi dalam **hanya 10-20 langkah evaluasi fungsi**.

### Konsep Dasar

#### Apa itu Diffusion Probabilistic Model?

Diffusion model adalah model generatif yang mempelajari proses denoising untuk menghasilkan data baru. Proses ini dapat dimodelkan sebagai persamaan diferensial biasa (ODE):

```
dx/dt = f(x, t)
```

dimana:
- `x` adalah data pada waktu `t`
- `f(x, t)` adalah fungsi drift yang memodelkan arah perubahan data

#### Perbedaan DPM-Solver vs DPM-Solver++

**DPM-Solver** (original):
- Menggunakan noise prediction model: `ε_θ(x_t, t)` → memprediksi noise yang ditambahkan
- Formulasi ODE dalam ruang noise

**DPM-Solver++** (improved):
- Menggunakan data prediction model: `x_θ(x_t, t)` → memprediksi data asli `x_0`
- Formulasi ODE yang lebih stabil, terutama untuk guided sampling
- Mengurangi error akumulasi pada guidance scale besar

#### Mengapa Multistep Order 2?

1. **Multistep**: Menggunakan informasi dari langkah-langkah sebelumnya untuk estimasi yang lebih akurat
   - Analog dengan metode Adams-Bashforth dalam numerik
   - Mengurangi jumlah evaluasi fungsi model yang diperlukan

2. **Order 2**: Menggunakan informasi dari 2 langkah sebelumnya
   - Balance antara akurasi dan efisiensi
   - Stabil untuk guided sampling dengan guidance scale besar
   - Lebih stabil daripada order 3 untuk guided sampling

---

## 🔬 Penjelasan Algoritma DPM-Solver++ 2M

### 1. Persamaan Dasar

Untuk diffusion model dengan noise schedule VP (Variance Preserving), forward process didefinisikan sebagai:

```
q(x_t | x_0) = N(α_t · x_0, σ_t² · I)
```

dimana:
- `α_t = √(α̂_t)` adalah koefisien mean
- `σ_t` adalah standar deviasi noise
- `λ_t = log(α_t) - log(σ_t)` adalah half-logSNR

### 2. Formulasi ODE untuk DPM-Solver++

DPM-Solver++ memformulasikan ODE dalam bentuk yang lebih stabil:

**Untuk data prediction model** (`x_θ` memprediksi `x_0`):

```
dx/dt = -σ_t' / σ_t · (x - α_t · x_θ(x_t, t))
```

dimana `'` menandakan turunan terhadap `t`.

### 3. Multistep DPM-Solver-2 Update

Dalam implementasi, `multistep_dpm_solver_second_update` melakukan update dari waktu `t_prev_0` ke `t` menggunakan informasi dari dua langkah sebelumnya:

```python
# Dari kode: dpm_solver_pytorch.py baris 796-852
def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
    """
    Update menggunakan:
    - model_prev_1: prediksi model pada waktu t_prev_1
    - model_prev_0: prediksi model pada waktu t_prev_0
    - x: nilai saat ini pada t_prev_0
    """
    
    # Hitung lambda values (half-logSNR)
    λ_prev_1 = λ(t_prev_1)
    λ_prev_0 = λ(t_prev_0)  
    λ_t = λ(t)
    
    # Hitung interval dalam ruang lambda
    h_0 = λ_prev_0 - λ_prev_1  # interval sebelumnya
    h = λ_t - λ_prev_0         # interval saat ini
    r0 = h_0 / h               # rasio interval
    
    # Estimasi turunan menggunakan finite difference
    D1_0 = (1 / r0) * (model_prev_0 - model_prev_1)
    
    # Update untuk DPM-Solver++
    φ_1 = expm1(-h)  # exp(-h) - 1
    x_t = (σ_t / σ_prev_0) * x - (α_t * φ_1) * model_prev_0 - 0.5 * (α_t * φ_1) * D1_0
```

**Penjelasan formula update:**

1. **Term pertama** `(σ_t / σ_prev_0) * x`: 
   - Skala nilai saat ini sesuai dengan perubahan noise schedule

2. **Term kedua** `(α_t * φ_1) * model_prev_0`:
   - Koreksi utama berdasarkan prediksi data pada langkah sebelumnya
   - `φ_1 = exp(-h) - 1` adalah koefisien dari ekspansi eksponensial

3. **Term ketiga** `0.5 * (α_t * φ_1) * D1_0`:
   - Koreksi orde kedua menggunakan estimasi turunan
   - `D1_0` adalah finite difference approximation dari turunan model
   - Koefisien `0.5` berasal dari ekspansi Taylor orde kedua

### 4. Algoritma Sampling Lengkap

```python
# Pseudocode untuk sampling dengan DPM-Solver++ 2M
def sample_dpm_solver_2m(x_T, steps=20):
    # Inisialisasi
    timesteps = generate_timesteps(steps)  # Membagi waktu menjadi steps interval
    x = x_T  # Mulai dari noise
    model_prev_list = []
    t_prev_list = []
    
    # Langkah 1: Inisialisasi dengan order 1 (DDIM)
    t = timesteps[0]
    model_prev_list.append(model_fn(x, t))
    t_prev_list.append(t)
    
    # Langkah 2: Update pertama dengan order 1
    t = timesteps[1]
    x = dpm_solver_first_update(x, t_prev_list[-1], t, model_prev_list[-1])
    model_prev_list.append(model_fn(x, t))
    t_prev_list.append(t)
    
    # Langkah 3 sampai akhir: Update dengan multistep order 2
    for step in range(2, steps + 1):
        t = timesteps[step]
        x = multistep_dpm_solver_second_update(
            x, 
            model_prev_list,  # Memakai 2 prediksi sebelumnya
            t_prev_list, 
            t
        )
        
        # Update history
        t_prev_list.append(t)
        model_prev_list.append(model_fn(x, t))
        
        # Hapus elemen lama jika diperlukan
        if len(model_prev_list) > 2:
            model_prev_list = model_prev_list[-2:]
            t_prev_list = t_prev_list[-2:]
    
    return x  # Data hasil denoising
```

### 5. Mengapa 2M Stabil untuk Guided Sampling?

1. **Data Prediction**: Memprediksi `x_0` langsung lebih stabil daripada memprediksi noise, terutama saat guidance scale besar

2. **Multistep**: Menggunakan informasi dari langkah sebelumnya mengurangi error akumulasi

3. **Order 2**: Balance optimal - order 1 terlalu lambat, order 3 bisa tidak stabil untuk guidance scale besar

4. **Koefisien Eksponensial**: Formula `φ_1 = exp(-h) - 1` mempertimbangkan struktur eksponensial dari noise schedule

---

## 💻 Implementasi dalam Kode

### Lokasi Implementasi Utama

**File**: `dpm_solver_pytorch.py`

1. **Kelas DPM_Solver** (baris 341-405):
    ```python
   class DPM_Solver:
       def __init__(self, model_fn, noise_schedule, algorithm_type="dpmsolver++"):
           # algorithm_type="dpmsolver++" untuk menggunakan data prediction
   ```

2. **Multistep Update Order 2** (baris 796-852):
   ```796:852:dpm_solver_pytorch.py
   def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
       """
       Multistep solver DPM-Solver-2 dari waktu t_prev_list[-1] ke waktu t.
       """
       # Implementasi formula di atas
   ```

3. **Sampling Function** (baris 1171-1203):
   ```1171:1203:dpm_solver_pytorch.py
   elif method == 'multistep':
       # Inisialisasi dengan order 1
       for step in range(1, order):
           # Update dengan lower order
       
       # Update dengan order 2 untuk langkah selanjutnya
       for step in range(order, steps + 1):
           x = self.multistep_dpm_solver_update(...)
   ```

### Cara Menggunakan DPM-Solver++ 2M

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# 1. Definisikan noise schedule
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# 2. Wrap model ke continuous-time noise prediction model
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type="noise",  # Model asli memprediksi noise
    model_kwargs=model_kwargs,
    guidance_type="classifier-free",  # Untuk guided sampling
    condition=condition,
    unconditional_condition=unconditional_condition,
    guidance_scale=guidance_scale,
)

# 3. Definisikan DPM-Solver dengan algorithm_type="dpmsolver++"
dpm_solver = DPM_Solver(
    model_fn, 
    noise_schedule, 
    algorithm_type="dpmsolver++"  # Menggunakan data prediction
)

# 4. Sample dengan multistep order 2
x_sample = dpm_solver.sample(
    x_T,                    # Noise awal
    steps=20,               # Jumlah langkah (10-20 sudah cukup)
    order=2,                # Order 2 untuk multistep
    skip_type="time_uniform",  # Uniform spacing
    method="multistep",     # Metode multistep
)
```

---

## 📊 Kapan Menggunakan DPM-Solver++ 2M?

### ✅ Direkomendasikan untuk:

1. **Guided Sampling dengan Guidance Scale Besar**
   - Classifier-free guidance dengan `guidance_scale > 5`
   - Classifier guidance
   - Stable Diffusion dengan prompt conditioning

2. **High-Resolution Images**
   - Resolusi ≥ 512x512
   - Lebih stabil daripada singlestep untuk high-res

3. **Latent-Space Diffusion Models**
   - Stable Diffusion
   - Model lain yang bekerja di latent space

### ⚠️ Alternatif untuk:

1. **Unconditional Sampling**
   - Lebih baik gunakan order 3 untuk kualitas lebih tinggi
   - Order 2 juga bisa bekerja, tapi order 3 lebih optimal

2. **Low-Resolution Images** (CIFAR-10, dll)
   - Gunakan `skip_type='logSNR'` daripada `'time_uniform'`
   - Order 2 tetap baik, tapi order 3 bisa lebih baik

3. **Pixel-Space Diffusion Models**
   - Pertimbangkan `correcting_x0_fn="dynamic_thresholding"` untuk kualitas lebih baik
   - Hanya untuk pixel-space, bukan latent-space

---

## 🎯 Parameter Penting

### Parameter Utama

| Parameter | Nilai Default | Penjelasan |
|-----------|---------------|------------|
| `algorithm_type` | `"dpmsolver++"` | Algoritma yang digunakan (++ untuk data prediction) |
| `order` | `2` | Order solver (2 untuk 2M) |
| `method` | `"multistep"` | Metode sampling (multistep untuk 2M) |
| `steps` | `20` | Jumlah evaluasi fungsi (10-20 cukup) |
| `skip_type` | `"time_uniform"` | Cara membagi timesteps |

### Rekomendasi Steps

- **Steps ≤ 10**: Sample cepat, kualitas cukup baik
- **Steps = 15**: Kualitas baik, balance optimal
- **Steps = 20**: Kualitas sangat baik (direkomendasikan)
- **Steps = 50**: Kualitas konvergen maksimal

### Skip Type

- **`"time_uniform"`**: Uniform dalam ruang waktu → **Direkomendasikan untuk high-res**
- **`"logSNR"`**: Uniform dalam ruang logSNR → Direkomendasikan untuk low-res
- **`"time_quadratic"`**: Quadratic spacing

---

## 📝 Contoh Lengkap: Stable Diffusion dengan DPM-Solver++ 2M

```python
import torch
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from diffusers import StableDiffusionPipeline

# Load model Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Definisikan noise schedule
betas = pipe.scheduler.betas  # Array beta dari scheduler
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# Wrap model untuk DPM-Solver
def model_fn(x, t_continuous):
    """Wrapper untuk model Stable Diffusion"""
    # Convert t_continuous ke discrete timestep
    timestep = (t_continuous - 1.0 / len(betas)) * len(betas)
    timestep = torch.clamp(timestep.long(), min=0, max=len(betas) - 1)
    
    # Panggil model
    with torch.no_grad():
        noise_pred = pipe.unet(x, timestep, encoder_hidden_states=text_embeddings).sample
    
    return noise_pred

# Definisikan DPM-Solver++ 2M
dpm_solver = DPM_Solver(
    model_fn,
    noise_schedule,
    algorithm_type="dpmsolver++"
)

# Encode prompt
prompt = "a beautiful landscape painting"
text_embeddings = pipe._encode_prompt(prompt, device="cuda", num_images_per_prompt=1)

# Sample dengan DPM-Solver++ 2M
latents = torch.randn((1, 4, 64, 64), device="cuda")
latents = latents * pipe.scheduler.init_noise_sigma

latents = dpm_solver.sample(
    latents,
    steps=20,
    order=2,
    skip_type="time_uniform",
    method="multistep",
)

# Decode ke gambar
image = pipe.vae.decode(latents / 0.18215).sample
image = (image / 2 + 0.5).clamp(0, 1)
```

---

## 🔍 Perbandingan dengan Metode Lain

| Metode | Steps | Kualitas | Waktu | Kestabilan Guidance |
|--------|-------|----------|-------|---------------------|
| **DDPM** | 1000 | ⭐⭐⭐⭐⭐ | 🐌🐌🐌 | ⭐⭐⭐ |
| **DDIM** | 50 | ⭐⭐⭐⭐ | 🐌🐌 | ⭐⭐⭐ |
| **DPM-Solver 1** | 20 | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐ |
| **DPM-Solver++ 1S** | 20 | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐ |
| **DPM-Solver++ 2M** | 20 | ⭐⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **DPM-Solver++ 3M** | 20 | ⭐⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐ |

**2M adalah pilihan terbaik untuk guided sampling!**

---

## 📚 Referensi dan Paper

### Paper Utama

1. **DPM-Solver**:
   - [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)
   - NeurIPS 2022 Oral

2. **DPM-Solver++**:
   - [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)

### Implementasi di Library Lain

- [🤗 Diffusers](https://github.com/huggingface/diffusers) - `DPMSolverMultistepScheduler`
- [Stable-Diffusion-WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)

---

## 🚀 Quick Start

### Instalasi

```bash
git clone https://github.com/LuChengTHU/dpm-solver.git
cd dpm-solver
```

Tidak perlu instalasi khusus - cukup import file `dpm_solver_pytorch.py` atau `dpm_solver_jax.py` ke proyek Anda!

### Contoh Minimal

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# Setup
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=your_betas)
model_fn = model_wrapper(your_model, noise_schedule, model_type="noise")
dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

# Sample
x_sample = dpm_solver.sample(
    x_T, steps=20, order=2, method="multistep"
)
```

---

## 📖 Dokumentasi Lebih Lanjut

Lihat bagian dokumentasi di bawah untuk:
- [Penjelasan detail noise schedule](#1-define-the-noise-schedule)
- [Cara wrap model yang berbeda](#2-wrap-your-model-to-a-continuous-time-noise-predicition-model)
- [Opsi sampling lainnya](#3-define-dpm-solver)
- [Contoh untuk guided sampling](#example-classifier-free-guidance-sampling-by-dpm-solver)

---

## 📄 License

Lihat file [LICENSE](LICENSE) untuk detail lisensi.

---

## 🙏 Citation

Jika Anda menggunakan kode ini dalam penelitian, mohon cite:

```bibtex
@article{lu2022dpm,
  title={DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2206.00927},
  year={2022}
}

@article{lu2022dpmplusplus,
  title={DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2211.01095},
  year={2022}
}
```

---

---

# 📚 Dokumentasi Teknis Lengkap

## 1. Definisi Noise Schedule

Kita mendukung noise schedule VP (Variance Preserving) untuk discrete-time dan continuous-time DPMs.

### 1.1. Discrete-time DPMs
  
Kita mendukung piecewise linear interpolation dari `log(α_t)` dalam kelas `NoiseScheduleVP`.

Kita memerlukan array `β_i` atau array `α̂_i` (lihat [DDPM](https://arxiv.org/abs/2006.11239) untuk detail). Perhatikan bahwa `α̂_i` dalam DDPM berbeda dari `α_t` dalam DPM-Solver:

```
α̂_i = ∏(1 - β_k)
α_{t_i} = √(α̂_i)
```

Definisikan discrete-time noise schedule dengan array `β_i`:
```python
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
```

Atau dengan array `α̂_i`:
```python
noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_cumprod)
```

### 1.2. Continuous-time DPMs

Kita mendukung linear schedule (seperti yang digunakan di DDPM dan ScoreSDE) dan cosine schedule (seperti yang digunakan di improved-DDPM).

Linear schedule:
```python
noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=0.1, continuous_beta_1=20.)
```

Cosine schedule:
```python
noise_schedule = NoiseScheduleVP(schedule='cosine')
```

---

## 2. Wrap Model ke Continuous-time Noise Prediction Model

Untuk model diffusion dengan input time label (bisa discrete-time labels atau continuous-time), kita wrap model function ke format berikut:

```python
model_fn(x, t_continuous) -> noise
```

dimana `t_continuous` adalah continuous time labels (0 sampai 1), dan output adalah noise prediction model.

### 2.1. Sampling Tanpa Guidance

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "noise" atau "x_start" atau "v" atau "score"
    model_kwargs=model_kwargs,
)
```

### 2.2. Sampling dengan Classifier Guidance

Untuk DPM dengan classifier guidance, kita perlu menentukan classifier function dan guidance scale:

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,
    model_kwargs=model_kwargs,
    guidance_type="classifier",
    condition=condition,
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    classifier_kwargs=classifier_kwargs,
)
```

### 2.3. Sampling dengan Classifier-Free Guidance

Untuk classifier-free guidance, model memerlukan input `cond` tambahan:

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,
    model_kwargs=model_kwargs,
    guidance_type="classifier-free",
    condition=condition,
    unconditional_condition=unconditional_condition,
    guidance_scale=guidance_scale,
)
```

---

## 3. Contoh Penggunaan Detail

### Contoh: Unconditional Sampling dengan DPM-Solver

Direkomendasikan menggunakan 3rd-order (dpmsolver atau dpmsolver++, multistep) DPM-Solver:

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# 1. Definisikan noise schedule
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# 2. Convert model ke continuous-time noise prediction model
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs=model_kwargs,
)

# 3. Definisikan dpm-solver
dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

# 4. Sample dengan singlestep DPM-Solver
x_sample = dpm_solver.sample(
    x_T,
    steps=20,
    order=3,
    skip_type="time_uniform",
    method="multistep",
)
```

### Contoh: Classifier-Free Guidance Sampling dengan DPM-Solver++ 2M

Direkomendasikan menggunakan 2nd-order (dpmsolver++, multistep) DPM-Solver:

```python
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# 1. Definisikan noise schedule
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# 2. Convert model ke continuous-time noise prediction model
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs=model_kwargs,
    guidance_type="classifier-free",
    condition=condition,
    unconditional_condition=unconditional_condition,
    guidance_scale=guidance_scale,
)

# 3. Definisikan dpm-solver dengan algorithm_type="dpmsolver++"
dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

# Untuk pixel-space DPMs, bisa tambahkan dynamic thresholding:
# dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
#                         correcting_x0_fn="dynamic_thresholding")

# 4. Sample dengan multistep DPM-Solver order 2
x_sample = dpm_solver.sample(
    x_T,
    steps=20,
    order=2,
    skip_type="time_uniform",
    method="multistep",
)
```

---

## 4. Tipe Model yang Didukung

| Tipe Model | Training Objective | Contoh Paper |
|-----------|-------------------|--------------|
| "noise": noise prediction model | `E[ω₁(t)||ε_θ(x_t,t) - ε||²]` | DDPM, Stable-Diffusion |
| "x_start": data prediction model | `E[ω₂(t)||x_θ(x_t,t) - x_0||²]` | DALL·E 2 |
| "v": velocity prediction model | `E[ω₃(t)||v_θ(x_t,t) - (α_tε - σ_tx_0)||²]` | Imagen Video |
| "score": marginal score function | `E[ω₄(t)||σ_t s_θ(x_t,t) + ε||²]` | ScoreSDE |

---

## 5. Metode Sampling yang Didukung

| Metode | Order yang Didukung | Thresholding | Keterangan |
|--------|---------------------|--------------|------------|
| DPM-Solver, singlestep | 1, 2, 3 | Tidak | Runge-Kutta-like |
| DPM-Solver, multistep | 1, 2, 3 | Tidak | Adams-Bashforth-like |
| DPM-Solver++, singlestep | 1, 2, 3 | Ya | |
| DPM-Solver++, multistep | 1, 2, 3 | Ya | **Direkomendasikan untuk guided sampling dengan order=2**, dan untuk unconditional sampling dengan order=3 |

---

## 6. Pengaturan Skip Type

Kita mendukung tiga tipe `skip_type` untuk pemilihan intermediate time steps:

- **`logSNR`**: Uniform logSNR untuk time steps. **Direkomendasikan untuk low-resolutional images**.
- **`time_uniform`**: Uniform time untuk time steps. **Direkomendasikan untuk high-resolutional images**.
- **`time_quadratic`**: Quadratic time untuk time steps.

---

## 7. Detail Implementasi Multistep DPM-Solver

Untuk multistep DPM-Solver dengan NFE = `steps`:

- **Order == 1**: Menggunakan K steps dari DPM-Solver-1 (DDIM).
- **Order == 2**: 
  - Pertama menggunakan 1 step dari DPM-Solver-1
  - Kemudian menggunakan (K-1) step dari multistep DPM-Solver-2
- **Order == 3**:
  - Pertama menggunakan 1 step dari DPM-Solver-1
  - Kemudian 1 step dari multistep DPM-Solver-2
  - Kemudian (K-2) step dari multistep DPM-Solver-3

---

## 8. Penggunaan dengan Diffusers Library

DPM-Solver++ 2M juga tersedia di library [🤗 Diffusers](https://github.com/huggingface/diffusers):

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Gunakan DPMSolverMultistepScheduler (DPM-Solver++)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
```

---

## 📝 Catatan Penting

1. **Kualitas Model**: Jika kualitas sampel dari 1000-step DDIM buruk, maka DPM-Solver tidak dapat memperbaikinya. DPM-Solver **dapat** mempercepat konvergensi, tapi **tidak dapat** memperbaiki kualitas sampel konvergen.

2. **Dynamic Thresholding**: Hanya valid untuk **pixel-space** diffusion models dengan `algorithm_type="dpmsolver++"`. **Tidak cocok** untuk latent-space models seperti Stable Diffusion.

3. **Denoise to Zero**: Dapat meningkatkan FID untuk low-resolutional images seperti CIFAR-10, tapi pengaruhnya kecil untuk high-resolutional images. Karena membutuhkan 1 evaluasi fungsi tambahan, tidak direkomendasikan untuk high-res images.

---

## 🔗 Referensi Tambahan

- [Repositori GitHub Resmi](https://github.com/LuChengTHU/dpm-solver)
- [Hugging Face Spaces Demo](https://huggingface.co/spaces/LuChengTHU/dpmsolver_sdm)
- [Paper DPM-Solver](https://arxiv.org/abs/2206.00927)
- [Paper DPM-Solver++](https://arxiv.org/abs/2211.01095)
