"""SQLite persistence for industrial visual inspection history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InspectionHistoryDB:
    """Stores and queries inspection results in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    confidence REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    backend TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    output_image TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def record(self, item: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    filename, status, severity, anomaly_score, threshold,
                    confidence, latency_ms, backend, model_version, output_image, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["filename"],
                    item["status"],
                    item["severity"],
                    item["anomaly_score"],
                    item["threshold"],
                    item["confidence"],
                    item["latency_ms"],
                    item["backend"],
                    item["model_version"],
                    item["output_image"],
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
