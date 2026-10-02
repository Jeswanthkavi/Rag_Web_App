# Backup & Disaster Recovery Procedures

## 1. Overview

The platform persists data across three stateful layers:
1. **Relational Data (MySQL)**: Users, documents metadata, chunk metadata, conversation history, and messages.
2. **Dense Vectors (Qdrant)**: Embeddings collections and HNSW vector index.
3. **Physical Files (Disk / Volume)**: Raw uploaded files (`.pdf`, `.docx`, `.txt`) in `./storage/uploads`.

---

## 2. Backup Procedures

### A. MySQL Database Backup
To take a full logical backup of the MySQL relational database:

```bash
# Export schema and data using mysqldump
mysqldump -h localhost -P 3306 -u rag_user -p advanced_rag > backup_mysql_$(date +%Y%m%d_%H%M%S).sql
```

If running inside Docker Compose:
```bash
docker compose exec -T mysql mysqldump -u root -p$MYSQL_ROOT_PASSWORD advanced_rag > backup_mysql_$(date +%Y%m%d_%H%M%S).sql
```

### B. Qdrant Vector Storage Backup
Qdrant supports creating point-in-time collection snapshots:

```bash
# Create snapshot for the rag_document_chunks collection
curl -X POST http://localhost:6333/collections/rag_document_chunks/snapshots
```
Snapshots are saved under `/qdrant/storage/snapshots/` and can be downloaded:
```bash
curl -O http://localhost:6333/collections/rag_document_chunks/snapshots/{snapshot_filename}
```

### C. Physical File Storage Backup
Archive the uploaded storage directory:
```bash
# Using tar to create an archive of all tenant upload folders
tar -czvf backup_uploads_$(date +%Y%m%d_%H%M%S).tar.gz ./storage/uploads/
```

If using Docker volumes:
```bash
docker run --rm -v advanced_rag_uploaded_docs:/volume -v $(pwd):/backup alpine tar -czvf /backup/backup_uploads_$(date +%Y%m%d_%H%M%S).tar.gz -C /volume .
```

---

## 3. Disaster Recovery / Restore Procedures

### A. Restore MySQL Database
1. Ensure the target database exists.
2. Import the backup SQL dump:
```bash
mysql -h localhost -P 3306 -u rag_user -p advanced_rag < backup_mysql_YYYYMMDD_HHMMSS.sql
```

### B. Restore Qdrant Vector Collection
1. Recover from snapshot via Qdrant REST API:
```bash
curl -X POST "http://localhost:6333/collections/rag_document_chunks/snapshots/recover" \
     -H "Content-Type: application/json" \
     -d '{"location": "http://backup-host/snapshots/rag_document_chunks_snapshot.snapshot"}'
```
2. Or use the built-in deterministic re-indexing script to re-embed and rebuild Qdrant from the MySQL chunks catalog if the snapshot is missing:
```bash
python scripts/verify_environment.py
```

### C. Restore Uploaded Files
Extract the archive back into the storage directory:
```bash
tar -xzvf backup_uploads_YYYYMMDD_HHMMSS.tar.gz -C ./storage/uploads/
```
Ensure directory permissions remain assigned to the application runner user (`appuser` UID 10001).
