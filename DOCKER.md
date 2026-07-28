# Shared Neo4j (Docker) — panduan

Neo4j jalan di container di PC ini. Tim akses lewat LAN. App (`main.py`) tetap
dijalankan dari venv lokal seperti biasa.

Alamat PC ini di LAN: **192.168.0.185** (bisa berubah kalau IP DHCP berganti —
cek ulang dengan `ipconfig getifaddr en1`).

---

## 1. Sekali di awal: matikan Neo4j lokal yang lama

Neo4j lokalmu sekarang memakai port 7687, jadi container bakal bentrok. Matikan dulu:

- Kalau lewat **Neo4j Desktop** → buka app → Stop database-nya.
- Kalau lewat **Homebrew** → `brew services stop neo4j`

Cek port sudah bebas: `lsof -iTCP:7687 -sTCP:LISTEN` (harus kosong).

## 2. Nyalakan Neo4j container

```bash
docker compose up -d
```

Pertama kali agak lama (download image + APOC). Lihat status/log:

```bash
docker compose ps
docker compose logs -f neo4j     # Ctrl-C buat keluar
```

Cek jalan: buka http://localhost:7474 → login pakai user/password dari `.env`.

## 3. Kamu build graph (dari venv lokal, seperti biasa)

`.env` nggak perlu diubah — `NEO4J_URI=neo4j://127.0.0.1:7687` tetap nyambung ke container.

```bash
python main.py --chunks chunks_data/maruf
```

## 4. Teman akses database (dari PC masing-masing, satu WiFi)

**Neo4j Browser (UI):**
1. Buka `http://192.168.0.185:7474`
2. Di kolom **Connect URL**, isi: `bolt://192.168.0.185:7687`
   (pakai `bolt://`, JANGAN `neo4j://` — untuk instance tunggal via LAN, `neo4j://`
   bisa gagal karena routing menunjuk balik ke localhost.)
3. Login pakai user/password yang sama.

**Dari kode (driver Python):**
```
NEO4J_URI=bolt://192.168.0.185:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password yang sama>
```

## 5. Firewall macOS

Kalau teman nggak bisa connect: System Settings → Network → Firewall. Kalau ON,
izinkan incoming connections untuk Docker, atau matikan sementara buat tes.
Pastikan juga semua di WiFi/subnet yang sama (192.168.0.x).

---

## Operasional

| Aksi | Perintah |
|---|---|
| Stop (data aman) | `docker compose stop` |
| Start lagi | `docker compose start` |
| Restart | `docker compose restart neo4j` |
| Lihat log | `docker compose logs -f neo4j` |
| Matikan + hapus container (data TETAP di volume) | `docker compose down` |
| ⚠️ Hapus SEMUA termasuk data graph | `docker compose down -v` |

- Data graph tersimpan di named volume `neo4j_data` — aman saat restart/`down`.
- Cuma `down -v` yang menghapus data. Hati-hati.
- Password: kalau container gagal start dan komplain soal password, ganti
  `NEO4J_PASSWORD` di `.env` jadi minimal 8 karakter, lalu `docker compose up -d` lagi.
