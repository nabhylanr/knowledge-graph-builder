# PROJECT_CONTEXT.md

> Dokumen konteks standalone untuk repo `knowledge-graph-builder`. Ditulis supaya
> bisa dipakai sebagai konteks di sesi Claude lain yang **tidak** punya akses ke
> filesystem repo ini. Fokus utama: `/src` (implementasi inti pembangunan
> Knowledge Graph). Bagian lain repo (`bench/`, `chunks_data/`, Docker) hanya
> disinggung sejauh perlu untuk memahami bagaimana `/src` dipanggil.

---

## Tentang project

Repo ini adalah implementasi teknis dari sebuah riset internship di lab
(pembimbing: Prof. Chou / Shintami Chusnul Hidayati), dengan tujuan membangun
sebuah **Knowledge Graph (KG) untuk knowledge management lab riset** —
khususnya untuk dua kebutuhan: **contradiction detection** (menemukan klaim
yang saling bertentangan antar dokumen) dan **claim history tracking**
(melacak evolusi sebuah klaim/topik dari waktu ke waktu, antar paper dan
meeting).

README repo menyebut proyek ini sebagai *"a stripped-down version of a larger
GraphRAG project"* — repo ini **tidak** punya UI (no Streamlit), **tidak**
melakukan document loading/cleaning/chunking, dan **tidak** punya
retrieval/Q&A. Ia murni menutupi satu potongan pipeline: **dari chunk teks
yang sudah jadi (`.jsonl`) sampai menjadi graph yang tersimpan di Neo4j**.
Proses menghasilkan chunk itu sendiri (loading dokumen mentah, OCR, chunking)
ditangani oleh proses/repo lain di luar cakupan ini — `chunks_data/` di root
repo ini sudah berisi hasil jadi dari proses tersebut, dikelompokkan per orang
per metode chunking (`wildan/`, `maruf/`, `linus/`).

**Siapa yang pakai:** tim intern lab yang sedang membandingkan beberapa metode
chunking untuk melihat mana yang menghasilkan KG terbaik (lihat
`bench/README.md`) sebelum commit ke satu metode untuk pipeline produksi.
Setup Neo4j di-share lewat Docker + LAN antar anggota tim (`DOCKER.md`).

**Catatan konteks penting** (bukan dari kode, tapi dari percakapan desain
sebelumnya dengan pemilik repo): proyek konseptualnya disebut *"Lab Brain"* /
`lab-brain-kg`, dengan skema ontologi awal berbasis node `Agent / Topic /
Metadata / Type / Description`. Kode aktual di `/src` (ontologi "v8", lihat di
bawah) sudah **berevolusi** dari desain awal itu — node yang di desain awal
disebut `Metadata` sekarang bernama `Source` di kode, dan kode menambahkan
node `Role` serta node `Contradiction` (reified conflict) yang di desain awal
masih berstatus rencana. Bagian **Status Implementasi** di bawah membedakan
mana yang sudah sinkron dengan rencana itu dan mana yang belum. (Sebagai
detail lucu yang mengonfirmasi keterkaitan dua sumber ini: prompt ekstraksi di
kode punya contoh literal "Prof. Chou" + "Shuo-Yan Chou" → "Prof. Shuo-Yan
Chou" sebagai aturan alias merging.)

---

## Arsitektur / skema inti (dari /src)

### Pipeline (lihat `main.py` sebagai orchestrator, logic ada di `/src`)

```
load chunks (.jsonl)  ->  embed  ->  extract graph (LLM, per chunk)  ->  sanitize (deterministik)  ->  store in Neo4j
                                                                                                          -> centralities & communities (opsional)
```

1. **Load** — `ChunksIngestor` (`src/ingestion/chunks_ingestor.py`) membaca
   `.jsonl`, mengelompokkan baris per `doc_id` menjadi `ProcessedDocument`.
2. **Embed** — `ChunkEmbedder` (`src/ingestion/embedder.py`) memberi setiap
   `Chunk.text` sebuah vector, lewat provider yang dipilih di `.env`.
3. **Extract** — `GraphMiner` (`src/ingestion/graph_miner.py`) memanggil
   `GraphExtractor` (`src/agents/graph_extractor.py`) satu kali per chunk: LLM
   mengisi *instance* graph mengikuti ontologi yang **fixed** di prompt
   (`src/prompts/graph_extractor.py`) — LLM tidak pernah mendefinisikan skema,
   hanya memilih node/edge mana yang berlaku untuk teks ini.
4. **Sanitize** — `sanitize_graph()` (`src/graph/graph_model.py`) menegakkan
   ulang ontologi itu secara **deterministik** (bukan trust ke output model):
   memperbaiki arah relationship, membuang self-loop, menggabungkan node
   Source/Topic duplikat, membatasi `has_source`, membuang apa pun di luar
   ontologi.
5. **Store** — `KnowledgeGraph` (`src/graph/knowledge_graph.py`, subclass dari
   `langchain_neo4j.Neo4jGraph`) menulis `Document`, `Chunk` (dengan
   embedding), node entitas, dan relationship struktural
   (`PART_OF`/`NEXT`/`MENTIONS`) ke Neo4j.
6. **Enrich** (opsional, `--no-communities` untuk skip) — `graph_ds.py`
   menghitung PageRank/betweenness/closeness dan mendeteksi komunitas
   Louvain & Leiden, lalu menulis hasilnya kembali sebagai property node.

### Model data (`src/schema.py`)

```python
class Chunk(BaseModel):
    chunk_id: Union[int, str]
    text: str
    filename: Optional[str]
    embedding: Optional[List[float]]
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embeddings_model: Optional[str]
    nodes: Optional[List[Node]]           # diisi setelah graph mining
    relationships: Optional[List[Relationship]]  # diisi setelah graph mining

class ProcessedDocument(BaseModel):
    filename: str
    source: str
    document_version: int = 1
    metadata: Optional[dict]
    chunks: Optional[List[Chunk]]
```

> **Catatan terbuka:** `schema.py` masih menyisakan definisi `Node` /
> `Relationship` / `Graph` yang di-comment-out di bagian bawah file. Kode aktif
> sudah memakai tipe `Node`/`Relationship` dari `langchain_neo4j.graphs.graph_document`
> untuk field ini, bukan class lokal itu. Tidak jelas dari kode apakah
> comment-out itu sisa migrasi yang lupa dibersihkan atau sengaja disimpan
> sebagai referensi — tidak diasumsikan salah satunya.

### Representasi graph internal saat ekstraksi (`src/graph/graph_model.py`)

Sebelum dipetakan ke tipe `langchain_neo4j`, hasil LLM disimpan dalam tipe
ringan lokal:

```python
class _Node(Serializable):        # id, type, properties
class _Relationship(Serializable): # source, target, type, properties
class _Graph(Serializable):        # nodes: List[_Node], relationships: List[_Relationship]
```

`sanitize_graph(_Graph) -> _Graph` beroperasi di level ini; baru di akhir,
`map_to_lc_graph()` mengonversi ke `GraphDocument` (tipe `langchain_neo4j`)
untuk ditulis ke Neo4j.

### Ontologi (fixed, "v8") — dari `src/prompts/graph_extractor.py` + `src/graph/graph_model.py`

**7 node label** (`ALLOWED_LABELS` di `graph_model.py`):

| Node | Makna | Property wajib |
|---|---|---|
| `Agent` | Orang/organisasi yang berkontribusi ke dokumen | `name` |
| `Role` | Fungsi Agent dalam konteks tertentu (Author, Supervisor, Speaker, ...) | `name` |
| `Topic` | Konsep/sistem/metode/metrik/section apa pun yang dibahas; punya hierarki lewat `has_subtopic` | `name` |
| `Type` | Kategori semantik tetap yang mengklasifikasi Topic (lihat taksonomi di bawah) | `name`, `domain` (auto: paper/meeting/unknown) |
| `Source` | Dokumen/file asal (satu node kanonik per dokumen) | `name`; opsional `format`, `date` |
| `Description` | Penjelasan tekstual kenapa satu Topic tergolong satu Type — unik per pasangan Topic×Type | `text`, `topicName`, `typeName` |
| `Contradiction` | Konflik tereifikasi antara ≥2 Description yang menyatakan fakta tak sejalan | `summary` (harus sebutkan detail konkret kedua sisi) |

**11 relationship name tetap** (`_FIXED_RELATION_DIRS` + 2 turunan `has_*`):

| Relationship | Arah | Catatan |
|---|---|---|
| `role_in_meeting` / `role_in_paper` | Agent → Role | |
| `spoke_about` / `writes_about` | Agent → Topic | optional `stance` ∈ {raised, proposed, decided, reported, gave_feedback} |
| `has_source` | Topic → Source | hanya 1–3 Topic top-level; **dibatasi maks 3 per dokumen** |
| `has_type` | Topic → Type | nama fixed — dulu dinamis per Type (`has_method`, dst), sekarang selalu `has_type` |
| `has_description` | Type → Description | nama fixed juga (dulu `has_[type]_description`) |
| `has_subtopic` | Topic → Topic | broader → narrower |
| `relates_to` | Topic → Topic | butuh `relation` dari controlled vocabulary + endpoint Type harus cocok tabel pasangan (lihat di bawah) |
| `assigned_to` | Topic → Agent | hanya kalau Topic ber-Type `Action Item` |
| `has_contradiction` | Description → Contradiction | butuh `level` ∈ {direct, partial, apparent} |

> **Catatan:** `Contradiction`/`has_contradiction` di dua tabel di atas
> menggambarkan konstanta `graph_model.py` (tidak diubah oleh penonaktifan
> STEP D — lihat prinsip #8 & Status Implementasi), **bukan** apa yang di-
> ekstrak per-chunk lagi (sekarang 6 node label / 10 relationship — lihat
> Status Implementasi). Skema `Contradiction` yang sebenarnya berlaku untuk
> whole-KB pass juga sudah berubah (tidak ada lagi `level`, diganti
> `resolution_type` dkk.) — lihat `docs/conflict_pipeline.md`, bukan
> `docs/conflict_ontology.md` yang sudah void.

Taksonomi **Type** (disjoint, 14 Paper + 7 Meeting):

- **Paper:** Background, Problem, Research Goal, Theoretical Basis, Dataset,
  Conclusion, Future Work, Existing Research, Research Gap, Method,
  Experiment, Result, Metrics Evaluation, Limitation.
- **Meeting:** Issue, Idea, Decision, Action Item, Open Question, Progress
  Update, Feedback.

`relates_to` punya 12 nilai `relation` yang masing-masing dibatasi pasangan
Type sumber→target (`RELATES_TO_TYPE_PAIRS` di `graph_model.py`), misalnya
`addresses` (Method → Problem|Research Goal), `evaluates` (Experiment|Metrics
Evaluation → Result), `contradicts` (Result → Result, atau Feedback →
Idea|Decision), dst. — daftar lengkap ada di kode/prompt.

Selain relationship yang diekstrak LLM, `KnowledgeGraph` menambahkan
relationship **struktural** (bukan hasil ekstraksi): `PART_OF` (Chunk →
Document), `NEXT` (urutan Chunk), `MENTIONS` (Chunk → entity apa pun yang
disebut di chunk itu), dan opsional `PRECEDES` (Document → Document, hanya
jika dokumen punya metadata `series` + `date`, untuk mengurutkan seri
meeting/versi paper secara kronologis).

### Ontologi konflik: Contradiction & Supersedes

> **Update:** `docs/conflict_ontology.md` (dirujuk di bagian ini) sudah
> **void**, digantikan `docs/conflict_pipeline.md` sebagai spec otoritatif.
> Per-chunk contradiction detection (STEP D) yang dulu disebut "aktif" di
> tabel bawah ini sudah **dinonaktifkan** — `Contradiction`/`has_contradiction`
> sekarang HANYA diproduksi oleh on-demand whole-KB pass (`src/conflict/`).

Dua bentuk hasil deteksi konflik, didefinisikan sebagai *single source of
truth* di `graph_model.py` dan dijelaskan penuh di `docs/conflict_pipeline.md`:

| Situasi | Output | Node baru? | Status |
|---|---|---|---|
| Dua fakta genuinely berkonflik, keduanya tetap berdiri | Node `Contradiction` + edge `has_contradiction` (≥2) | Ya | **Hanya** dari whole-KB pass on-demand — per-chunk detection dinonaktifkan (lihat Status Implementasi untuk progres pass itu sendiri) |
| Result yang lebih baru mengoreksi Result lama | Edge `supersedes` (Description → Description, arah baru→lama) | Tidak — edge saja | Idem — produk pass on-demand yang sama |

`supersedes` sengaja **tidak** dimasukkan ke peta ontologi construction
(`_FIXED_RELATION_DIRS`) karena ia hanya diproduksi oleh pass on-demand
terpisah yang melihat seluruh KB, bukan oleh ekstraksi per-chunk.

---

## Prinsip desain kunci

Daftar berikut fokus pada keputusan yang **sudah dipertimbangkan opsi
lainnya** — supaya pembaca lain tidak mengusulkan ulang sesuatu yang sudah ada
alasan penolakannya.

1. **Ontologi fixed di prompt, tidak configurable per-instance.** Dulu ada
   parameter `ontology: Optional[Ontology]` yang di-thread dari
   `Configuration` ke `GraphExtractor`, tapi ternyata **tidak pernah dipakai**
   (silently discarded) — sudah dihapus. Prinsipnya: LLM hanya mengisi
   *instance*, tidak pernah mendefinisikan skema; skema hidup di
   `src/prompts/graph_extractor.py` dan ditegakkan ulang oleh `sanitize_graph`.

2. **Nama relationship untuk `has_[type]` dibuat FIXED (`has_type` /
   `has_description`), bukan dinamis per Type** (`has_method`, `has_decision`,
   dst — ini perilaku versi sebelumnya, "v7"). Alasan: nama dinamis membuat
   relationship-schema unbounded tanpa menambah informasi apa pun, karena id
   node `Type` itu sendiri sudah meng-encode kategori yang sama. Prinsip yang
   sama diterapkan ulang untuk `has_contradiction`'s `level` — level **selalu**
   jadi property, bukan bagian nama edge (`contradict_level1` ditolak
   eksplisit di bagian "MISTAKES" prompt).

3. **`relates_to` divalidasi lewat tabel pasangan Type, bukan cuma prosa.**
   `RELATES_TO_TYPE_PAIRS` membuat `sanitize_graph` bisa menolak edge yang
   `relation`-nya string valid (mis. "uses") tapi endpoint Type-nya salah
   (mis. dari Decision, bukan Method/Experiment).

4. **Topic dedup berlapis dua, dan sengaja BERHENTI sebelum semantic
   matching:**
   - `_resolve_abbreviation_aliases` — merge node singkatan bare ke nama
     penuhnya, tapi HANYA jika node penuh punya property `abbreviation` yang
     eksplisit cocok.
   - `_dedupe_similar_topics` — merge string near-identik (typo/plural varian)
     via `difflib.SequenceMatcher`, threshold **0.92**.
   - **Sengaja tidak** melakukan semantic synonym merge (mis. "Digital Twin"
     vs "Digital Twin System") di level ini. Alasan tertulis di kode: di atas
     ~0.9 similarity, false-positive merge (diam-diam menggabungkan dua Topic
     yang sebenarnya beda) menjadi lebih costly daripada duplikat yang mau
     dicegah — itu perlu **embedding-based matching + human review**, yang
     sengaja dianggap out of scope untuk sanitizer deterministik ini.
   - **Catatan terbuka:** ini persis titik yang di desain konseptual awal
     proyek direncanakan pakai *embedding similarity check saat ingest* untuk
     resolusi Topic. Di kode saat ini, dedup Topic murni string-based —
     embedding belum dipakai untuk tujuan itu (lihat Status Implementasi).

5. **`has_source` dibatasi maksimal 3 edge per Source per DOKUMEN, bukan per
   chunk.** Dilacak lewat `has_source_state`, sebuah dict yang dibuat **sekali**
   oleh `GraphMiner` per dokumen dan di-passing ke setiap panggilan
   `sanitize_graph` untuk tiap chunk dokumen itu. Kalau cap direset per chunk,
   dokumen N-chunk bisa berakhir dengan sampai 3×N edge `has_source`.

6. **Syarat minimal 2 partisipan untuk `Contradiction` sengaja TIDAK
   ditegakkan di `sanitize_graph`.** Alasannya: `sanitize_graph` hanya melihat
   satu chunk per pemanggilan, sementara partisipan ke-2 sebuah Contradiction
   mungkin baru muncul di chunk berikutnya — men-drop node itu secara eager
   akan membuang edge pertama yang sudah ter-persist. Solusinya dipindah jadi
   query pembersihan pasca-ingest (`KnowledgeGraph._cleanup_singleton_contradictions`)
   yang jalan sekali setelah SEMUA chunk dokumen tersimpan, menghapus
   `Contradiction` dengan <2 edge `has_contradiction` dari graph Neo4j yang
   sesungguhnya (bukan dari view per-chunk yang parsial).

7. **`supersedes` sengaja edge-only, tanpa node** — beda dari `Contradiction`
   yang direifikasi jadi node. Alasan (dari komentar kode): "supersedes
   basically just updates information" — bukan konflik genuine yang perlu
   diklasifikasi (tidak ada `level`). Constraint-nya: kedua endpoint **wajib**
   `Description` dengan `typeName == "Result"` (hanya findings/results yang
   "dikoreksi", bukan Background/Method/dst), tidak boleh self-loop, dan
   **anti-cycle** — `A supersedes B` DAN `B supersedes A` tidak boleh
   coexist (itu seharusnya jadi `Contradiction`, bukan supersession).

8. **Deteksi konflik per-chunk (STEP D) sudah DINONAKTIFKAN — whole-KB pass
   sekarang satu-satunya producer.** Sebelumnya per-chunk (prompt STEP D +
   edge `has_contradiction`) hanya bisa menangkap konflik antar-Description
   yang **sedang ditulis di output yang sama**, tidak bisa lintas-chunk/dokumen
   — lihat `docs/conflict_pipeline.md` §"Sole ownership of Contradiction"
   untuk alasan penonaktifannya. Pass whole-KB (spec penuh di
   `docs/conflict_pipeline.md`, menggantikan `docs/conflict_ontology.md` yang
   sudah void) dirancang lewat: **blocking** (structural + kNN) →
   **gates** (cheap rejection) → **classification** (LLM, lazy) — progres
   implementasinya di Status Implementasi.

9. **Sanitizer sebagai *safety net* deterministik, bukan pengganti prompt
   engineering — dan sebaliknya.** Komentar eksplisit di prompt ("HONEST
   NOTE"): prompt yang lebih baik mengurangi banyak masalah, tapi model
   sekecil `llama-3.1-8b-instant` tetap bisa "leak" constraint numerik atau
   self-loop; `sanitize_graph` yang menegakkan itu secara deterministik —
   jangan terus menulis ulang prompt untuk mengejar kasus itu.

10. **Ekstraksi per-chunk paralel, sanitize+map serial (order-dependent).**
    `GraphMiner` menjalankan ekstraksi LLM tiap chunk lewat `ThreadPoolExecutor`
    (`EXTRACTOR_MAX_WORKERS`, default 4 — 1 = sequential penuh), tapi tahap
    sanitize+map berikutnya sengaja dijalankan **serial** dalam urutan chunk
    asli, karena `sanitize_graph` punya state lintas-chunk yang
    order-dependent (`has_source_state`: cap 3-pertama; `topic_registry`:
    dedup ejaan-pertama). Memparalelkan tahap ini akan membuat output
    non-deterministik antar-run.

11. **Fallback raw-JSON parsing untuk model kecil.** Banyak model (qwen3-vl,
    llama-3.1-8b) gagal di tool/function calling (`with_structured_output`
    balik graph kosong) tapi tetap menghasilkan JSON valid sebagai teks biasa.
    `GraphExtractor.extract_graph` fallback ke `_parse_graph_from_text` kalau
    structured-output kosong ATAU error-nya bukan rate-limit. Env
    `EXTRACTOR_RAW_ONLY=1` men-skip structured-output sepenuhnya untuk model
    yang konsisten gagal di situ (mis. qwen3-vl) — menghemat 1 LLM call per
    chunk.

12. **Retry logic membedakan 3 kelas error**, bukan retry generik: rate limit
    (429 → tunggu 65s, kecuali pesan "limit_exceeded ... limit: 0" yang
    dianggap permanent dan tidak di-retry), transient connection error (server
    disconnect/refused/reset dll → tunggu 30s) yang **ditambahkan setelah
    insiden konkret**: satu server Ollama remote yang blip ~15 menit diam-diam
    menjatuhkan ~50 dari 106 chunk kalau tidak di-retry.

13. **Domain tagging Type (`paper`/`meeting`/`unknown`) dilakukan
    deterministically**, bukan trust ke output model — dan vocabulary
    Paper/Meeting sengaja dibuat **disjoint** di v8 (v7 sebelumnya punya
    bucket "shared"), supaya cukup lihat satu label Type untuk tahu domain
    asalnya; "unknown" hanya muncul kalau model invent Type baru yang tidak
    cocok keduanya.

14. **Endpoint LLM/embedding remote opsional untuk Ollama** (via
    `RE_MODEL_ENDPOINT` / `EMBEDDINGS_ENDPOINT`), supaya laptop yang lemah bisa
    offload inference ke server lain (mis. lewat Tailscale). Default tetap
    `localhost:11434` kalau env unset atau `"none"`.

---

## Status implementasi

### ✅ Selesai (jalan, sudah di-commit)

- Load chunks dari `.jsonl` — file tunggal atau folder (`ChunksIngestor`).
- Embedding chunk text — provider Ollama/OpenAI/Azure/HuggingFace
  (`ChunkEmbedder` + `factory/embeddings.py`).
- Ekstraksi graph per-chunk lewat LLM dengan ontologi v8 penuh (6 node label,
  10 relationship — `Contradiction`/`has_contradiction` TIDAK lagi bagian dari
  ekstraksi per-chunk, lihat prinsip #8) — prompt rinci dengan self-check
  checklist 9 poin (`GraphExtractor` + `prompts/graph_extractor.py`).
- Sanitizer deterministik lengkap (`sanitize_graph`) yang menegakkan semua
  constraint di bagian Prinsip Desain di atas.
- Penyimpanan ke Neo4j: node `Document`/`Chunk`, edge `PART_OF`/`NEXT`/`MENTIONS`,
  vector index untuk `Chunk` (`KnowledgeGraph`).
- Edge `PRECEDES` antar `Document` (opt-in, perlu metadata `series` + `date`).
- Cleanup `Contradiction` singleton pasca-ingest (sekarang praktis no-op sejak
  per-chunk detection dinonaktifkan, tapi tetap dipertahankan sebagai safety net).
- Centralities (PageRank, betweenness, closeness) + community detection
  (Louvain & Leiden) + modularity, ditulis kembali sebagai property node /
  node `GraphMetric` (`graph_ds.py`).
- CLI entrypoint `main.py` (`--chunks`, `--limit`, `--no-communities`).
- Setup Neo4j via Docker untuk shared dev antar tim lewat LAN
  (`docker-compose.yml`, `DOCKER.md`) — di luar `/src`, tapi jadi jalur
  deployment yang sudah jadi.

### 🔶 Sebagian / baru spesifikasi (kode belum lengkap menjalankannya)

- **`supersedes` edge:** spesifikasi lengkap ada di `docs/conflict_pipeline.md`
  (menggantikan `docs/conflict_ontology.md` yang sudah void), dan konstanta
  terkait sudah ada di `graph_model.py`. Eksekusi (classification/write-back)
  masih berjalan — lihat status blocking/gates di bawah.
- **`Contradiction` node:** per-chunk detection (STEP D) sudah **dinonaktifkan**
  (lihat prinsip #8) — bukan lagi "sebagian jalan", sekarang murni TIDAK
  diproduksi oleh ekstraksi sama sekali. Satu-satunya jalur produksi adalah
  whole-KB pass on-demand.

### ⛔ Belum dimulai

- **Classification & write-back** untuk whole-KB conflict pass (§4 di
  `docs/conflict_pipeline.md`) — blocking (`src/conflict/blocking.py`) dan
  gates (`src/conflict/gates.py`) sudah ada, tapi belum ada kode yang
  benar-benar menulis `Contradiction`/`supersedes` ke Neo4j.
- **Topic resolution via embedding similarity saat ingest.** Saat ini dedup
  Topic murni string-based (`difflib`, threshold 0.92 — lihat prinsip #4).
  Ada dua catatan eksplisit di kode yang menyebut "embedding-based matching"
  sebagai out of scope untuk implementasi saat ini
  (`graph_model.py:355`, `prompts/graph_extractor.py:61`) — jadi ini bukan
  cuma belum dikerjakan, tapi sudah didokumentasikan sebagai gap yang
  disadari.
- **Layer query/retrieval untuk claim history** (traversal Topic → Type →
  Description terurut by timestamp) atau untuk "isi dari Source X" — tidak
  ada kode query semacam ini di `/src`. README repo ini eksplisit menyatakan
  repo hanya menutupi construction pipeline, "no retrieval/Q&A".

---

## Integrasi dengan bagian lain repo

- **`main.py`** (root) — satu-satunya entrypoint yang memanggil `/src`. Alur:
  baca `.env` (`python-dotenv`) → `build_configuration()` merakit
  `Configuration` (pydantic) dari env vars → `ChunksIngestor` →
  `ChunkEmbedder` → `GraphMiner` → `KnowledgeGraph.add_documents()` →
  opsional `update_centralities_and_communities()`.
- **`chunks_data/`** (root) — dataset `.jsonl` siap pakai, dikelompokkan per
  orang/metode chunking (`wildan/`, `maruf/`, `linus/`). `/src` **tidak**
  melakukan chunking sendiri; ia murni mengonsumsi file ini lewat
  `ChunksIngestor`.
- **`bench/`** (root, working directory saat percakapan ini berlangsung) —
  tooling terpisah untuk membandingkan metode chunking, **konsumen** dari
  `/src` (bukan bagian dari pipeline produksi): `prep.py` menormalkan dua
  format native jadi satu skema `.jsonl`; `chunk_metrics.py` menghitung
  metrik intrinsik chunk (tanpa LLM); `graph_metrics.py` memanggil
  `GraphMiner`/`GraphExtractor`/`sanitize_graph` dari `/src` untuk membangun
  graph in-memory (via `networkx`, tanpa Neo4j) demi skoring struktur graph;
  `eval_gold.py` mengukur recall terhadap gold set manual di `bench/gold/`.
- **`docker-compose.yml` + `DOCKER.md`** (root) — infra Neo4j via Docker untuk
  sharing antar tim lewat LAN; tidak mengubah cara `/src` dipanggil (`main.py`
  tetap jalan dari venv lokal, hanya nyambung ke `NEO4J_URI` container yang
  sama).
- **`.env` / `.env_example`** (root) — satu-satunya jalur konfigurasi masuk ke
  `/src` (lewat `os.getenv` di `main.py` → model `Configuration` pydantic):
  kredensial Neo4j, tipe/model LLM ekstraksi, tipe/model embedding, dan 3 env
  knob kecepatan ekstraksi (`EXTRACTOR_RAW_ONLY`, `EXTRACTOR_MAX_WORKERS`,
  `EXTRACTOR_NO_THINK`).

---

## Stack & dependency

Dipakai langsung oleh `/src` (lihat `requirements.txt` di root untuk versi
persis):

| Library | Untuk apa di `/src` |
|---|---|
| `pydantic` 2.10 | Semua config & schema model (`config.py`, `schema.py`) |
| `langchain` / `langchain-core` | `PromptTemplate` untuk prompt ekstraksi; tipe dasar `Document` |
| `langchain-neo4j` | `Neo4jGraph` (base class `KnowledgeGraph`), `Neo4jVector` (vector store untuk `Chunk`), tipe `Node`/`Relationship`/`GraphDocument` |
| `neo4j` (driver resmi) | Dipakai langsung via `self._driver.session()` untuk query custom (update property, export ke `networkx`, dll) di luar yang disediakan `Neo4jGraph` |
| `langchain-ollama` | Provider LLM/embedding default untuk SEMUA stage — ekstraksi, gate NLI, classification (`qwen3:4b` / `mxbai-embed-large`), dengan dukungan endpoint remote |
| `langchain-openai` | Provider OpenAI + Azure OpenAI (LLM & embedding) |
| `langchain-google-genai` | Provider Google Gemini (opsional) |
| `langchain-huggingface` | Provider HuggingFace transformers — **lazy-imported**, hanya kalau `type="trf"`, supaya run Ollama tidak ikut menarik dependency `torch`/`transformers` (~GB) |
| `networkx` | Representasi graph in-memory (`KnowledgeGraph.get_digraph()`, dipakai juga oleh `bench/graph_metrics.py`); PageRank/betweenness/closeness |
| `python-louvain` (import sebagai `community`) | Deteksi komunitas Louvain |
| `python-igraph` + `leidenalg` | Deteksi komunitas Leiden |
| `python-dotenv` | Load `.env` di `main.py` |

---

## Struktur folder /src

```
src/
├── __init__.py
├── config.py                 # pydantic config models: ModelType, LLMConf,
│                              # EmbedderConf, KnowledgeGraphConfig, Configuration.
│                              # Docstring menegaskan: ontologi TIDAK dikonfigurasi di sini.
├── schema.py                  # Chunk & ProcessedDocument (model domain pipeline);
│                              # menyisakan definisi Node/Relationship/Graph lama
│                              # yang di-comment-out (lihat catatan terbuka di atas)
│
├── factory/                   # Resolusi provider LLM & embedding dari config
│   ├── __init__.py
│   ├── llm.py                 # fetch_llm(conf) -> BaseChatModel
│   │                          # (Ollama/OpenAI/Azure/Google/HF, lazy-import utk Google & HF)
│   └── embeddings.py          # get_embeddings(conf) -> Embeddings
│                              # (Ollama/OpenAI/Azure/HF, lazy-import utk HF)
│
├── ingestion/                  # Baca chunk siap-pakai -> embed -> mine graph
│   ├── __init__.py
│   ├── chunks_ingestor.py     # ChunksIngestor: baca .jsonl -> group by doc_id
│   │                          # -> List[ProcessedDocument]
│   ├── embedder.py             # ChunkEmbedder: embed tiap Chunk.text
│   └── graph_miner.py          # GraphMiner: orkestrasi per-dokumen —
│                              # ekstraksi PARALEL per chunk, sanitize+map SERIAL
│                              # (state lintas-chunk order-dependent)
│
├── agents/
│   ├── __init__.py
│   └── graph_extractor.py     # GraphExtractor: satu LLM call per chunk,
│                              # + retry (rate-limit/connection error) +
│                              # fallback raw-JSON parsing utk model kecil
│
├── prompts/
│   ├── __init__.py
│   └── graph_extractor.py     # get_graph_extractor_prompt(): PromptTemplate
│                              # berisi SELURUH ontologi v8 (node types,
│                              # relationship table, vocabulary, contoh JSON,
│                              # 11-poin self-check checklist). ~460 baris —
│                              # salah satu file paling padat di /src.
│
├── graph/                      # Submodul TERBESAR & paling kompleks di /src
│   ├── __init__.py
│   ├── graph_model.py          # _Node/_Relationship/_Graph (representasi internal),
│   │                          # semua konstanta ontologi (ALLOWED_LABELS,
│   │                          # _FIXED_RELATION_DIRS, RELATES_TO_TYPE_PAIRS,
│   │                          # SUPERSEDES_*, dll), sanitize_graph() — jantung
│   │                          # deterministik seluruh project (~670 baris),
│   │                          # + map_to_lc_*() converter ke langchain_neo4j
│   ├── knowledge_graph.py      # KnowledgeGraph(Neo4jGraph): semua operasi Cypher
│   │                          # (Document/Chunk/PART_OF/NEXT/MENTIONS/PRECEDES,
│   │                          # cleanup Contradiction singleton, property
│   │                          # statistik graph, update centralities/communities)
│   └── graph_ds.py              # Fungsi standalone berbasis networkx/igraph/leidenalg:
│                              # detect_louvain_communities, detect_leiden_communities,
│                              # compute_centralities, build_update_query,
│                              # update_modularity — dipanggil dari KnowledgeGraph
│
└── utils/
    ├── __init__.py
    └── logger.py               # get_logger() (console + file handler `app.log`),
                                 # disable_logger() context manager
```

Submodul paling besar/kompleks: **`graph/`** (terutama `graph_model.py`, mesin
sanitasi ontologi) dan **`prompts/graph_extractor.py`** (spesifikasi ontologi
lengkap dalam bentuk prompt teks). Keduanya harus dibaca berpasangan — prompt
mendefinisikan apa yang *diminta* dari LLM, `graph_model.py` mendefinisikan apa
yang *ditegakkan* terlepas dari yang benar-benar dikembalikan LLM.
