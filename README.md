<!-- Don't delete it -->
<div name="readme-top"></div>

<!-- Organization Logo -->
<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 16px;">
  <img alt="AOSSIE" src="public/aossie-logo.svg" width="175">
  <img src="public/todo-project-logo.svg" width="175" />
</div>

&nbsp;

<!-- Organization Name -->
<div align="center">

[![Static Badge](https://img.shields.io/badge/aossie.org/TODO-228B22?style=for-the-badge&labelColor=FFC517)](https://TODO.aossie.org/)

<!-- Correct deployed url to be added -->

</div>

<!-- Organization/Project Social Handles -->
<p align="center">
<!-- Telegram -->
<a href="https://t.me/StabilityNexus">
<img src="https://img.shields.io/badge/Telegram-black?style=flat&logo=telegram&logoColor=white&logoSize=auto&color=24A1DE" alt="Telegram Badge"/></a>
&nbsp;&nbsp;
<!-- X (formerly Twitter) -->
<a href="https://x.com/aossie_org">
<img src="https://img.shields.io/twitter/follow/aossie_org" alt="X (formerly Twitter) Badge"/></a>
&nbsp;&nbsp;
<!-- Discord -->
<a href="https://discord.gg/hjUhu33uAn">
<img src="https://img.shields.io/discord/1022871757289422898?style=flat&logo=discord&logoColor=white&logoSize=auto&label=Discord&labelColor=5865F2&color=57F287" alt="Discord Badge"/></a>
&nbsp;&nbsp;
<!-- Medium -->
<a href="https://news.stability.nexus/">
  <img src="https://img.shields.io/badge/Medium-black?style=flat&logo=medium&logoColor=black&logoSize=auto&color=white" alt="Medium Badge"></a>
&nbsp;&nbsp;
<!-- LinkedIn -->
<a href="https://www.linkedin.com/company/aossie/">
  <img src="https://img.shields.io/badge/LinkedIn-black?style=flat&logo=LinkedIn&logoColor=white&logoSize=auto&color=0A66C2" alt="LinkedIn Badge"></a>
&nbsp;&nbsp;
<!-- Youtube -->
<a href="https://www.youtube.com/@StabilityNexus">
  <img src="https://img.shields.io/youtube/channel/subscribers/UCZOG4YhFQdlGaLugr_e5BKw?style=flat&logo=youtube&logoColor=white&logoSize=auto&labelColor=FF0000&color=FF0000" alt="Youtube Badge"></a>
</p>

---

<div align="center">
<h1>OpenVerifiableLLM – Deterministic Dataset Pipeline</h1>
</div>

OpenVerifiableLLM is a deterministic Wikipedia preprocessing and dataset verification pipeline designed to support fully reproducible LLM training.

It ensures that:

- The same Wikipedia dump always produces identical processed output.
- Dataset fingerprints (SHA256 hashes) are generated for verification.
- A manifest file captures dataset identity and environment metadata.

---

## 🚀 Features

- **Deterministic Wikipedia preprocessing**
- **Wikitext cleaning (templates, references, links removed)**
- **Stable XML parsing with memory-efficient streaming**
- **SHA256 hashing of raw and processed datasets**
- **Automatic dataset manifest generation**
- **Reproducible data identity tracking**

---

## 💻 Tech Stack

- Python 3.9+
- `xml.etree.ElementTree` (stream parsing)
- `bz2` (compressed dump handling)
- `hashlib` (SHA256 hashing)
- `pathlib`
- `re` (deterministic cleaning)

---

## 📂 Project Structure

```text
OpenVerifiableLLM/
│
├── scripts/
│ ├── preprocess.py
│ ├── generate_manifest.py
│ └── hash_utils.py
│
├── data/
│ ├── raw/
│ └── processed/
│
└── dataset_manifest.json
```

---

## �🍀 Getting Started

### Prerequisites

- Python 3.9+
- Wikipedia dump from:
  https://dumps.wikimedia.org/

Recommended for testing:
- `simplewiki-YYYYMMDD-pages-articles.xml.bz2`

---

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/AOSSIE-Org/OpenVerifiableLLM.git
cd OpenVerifiableLLM
```

### ▶ Running the Pipeline

#### Step 1 — Place Dump File
Move your Wikipedia dump into:

```bash
data/raw/
```

Example:

```bash
data/raw/simplewiki-20260201-pages-articles.xml.bz2
```
#### Step 2 — Run Preprocessing

```bash
python scripts/preprocess.py data/raw/simplewiki-20260201-pages-articles.xml.bz2
```
This will:
- Create `data/processed/wiki_clean.txt`
- Generate `dataset_manifest.json`
- Compute `SHA256` hashes

#### 📜 Example Manifest

```json
{
  "wikipedia_dump": "simplewiki-20260201-pages-articles.xml.bz2",
  "dump_date": "2026-02-01",
  "raw_sha256": "...",
  "processed_sha256": "...",
  "preprocessing_version": "v1",
  "python_version": "3.13.2"
}
```
## 📈 Future Extensions
- Deterministic tokenization stage
- Token-level hashing
- Multi-GPU training reproducibility
- Environment containerization (Docker)
- Full checkpoint verification protocol

---

## 📱 App Screenshots

TODO: Add screenshots showcasing your application

|  |  |  |
|---|---|---|
| Screenshot 1 | Screenshot 2 | Screenshot 3 |

---

## 🙌 Contributing

⭐ Don't forget to star this repository if you find it useful! ⭐

Thank you for considering contributing to this project! Contributions are highly appreciated and welcomed. To ensure smooth collaboration, please refer to our [Contribution Guidelines](./CONTRIBUTING.md).

---

## ✨ Maintainers

TODO: Add maintainer information

- [Maintainer Name](https://github.com/username)
- [Maintainer Name](https://github.com/username)

---

## 📍 License

This project is licensed under the GNU General Public License v3.0.
See the [LICENSE](LICENSE) file for details.

---

## 💪 Thanks To All Contributors

Thanks a lot for spending your time helping TODO grow. Keep rocking 🥂

[![Contributors](https://contrib.rocks/image?repo=AOSSIE-Org/TODO)](https://github.com/AOSSIE-Org/TODO/graphs/contributors)

© 2025 AOSSIE 
