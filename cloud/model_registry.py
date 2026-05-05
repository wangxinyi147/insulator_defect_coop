# cloud/model_registry.py
"""SQLite 模型注册表 - 记录每次训练的指标，支持模型版本管理"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path


class ModelRegistry:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(__file__).parent / "model_registry.db"
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT,
                model_path TEXT,
                mAP50 REAL,
                mAP50_95 REAL,
                precision REAL,
                recall REAL,
                flashover_map50 REAL,
                flashover_recall REAL,
                lose_map50 REAL,
                damaged_map50 REAL,
                attention_module TEXT,
                lightweight BOOLEAN DEFAULT 0,
                onnx_path TEXT,
                trt_path TEXT,
                onnx_size_mb REAL,
                params_m REAL,
                epochs INTEGER,
                batch_size INTEGER,
                imgsz INTEGER,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'running',
                log_file TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS feedback_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_received INTEGER DEFAULT 0,
                false_positive INTEGER DEFAULT 0,
                false_negative INTEGER DEFAULT 0,
                used_for_training INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()

    def register(self, metrics_dict, file_paths, cfg_info=None):
        """注册新训练的模型"""
        if self.conn is None:
            return None

        # 取消其他模型的激活状态
        self.conn.execute("UPDATE models SET is_active = 0")

        self.conn.execute("""
            INSERT INTO models (
                name, version, model_path,
                mAP50, mAP50_95, precision, recall,
                flashover_map50, flashover_recall,
                lose_map50, damaged_map50,
                attention_module, lightweight,
                onnx_path, trt_path,
                epochs, batch_size, imgsz,
                is_active, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (
            metrics_dict.get("name", "insulator_defect"),
            metrics_dict.get("version", datetime.now().strftime("%Y%m%d_%H%M%S")),
            file_paths.get("pt_path", ""),
            metrics_dict.get("mAP50", 0),
            metrics_dict.get("mAP50_95", 0),
            metrics_dict.get("precision", 0),
            metrics_dict.get("recall", 0),
            metrics_dict.get("flashover_map50", 0),
            metrics_dict.get("flashover_recall", 0),
            metrics_dict.get("lose_map50", 0),
            metrics_dict.get("damaged_map50", 0),
            cfg_info.get("attention_module", "None") if cfg_info else "None",
            1 if (cfg_info and cfg_info.get("lightweight_enhance")) else 0,
            file_paths.get("onnx_path", ""),
            file_paths.get("trt_path", ""),
            cfg_info.get("epochs", 100) if cfg_info else 100,
            cfg_info.get("batch_size", 8) if cfg_info else 8,
            cfg_info.get("imgsz", 640) if cfg_info else 640,
            metrics_dict.get("notes", ""),
        ))
        self.conn.commit()
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_active(self):
        """获取当前激活的模型"""
        row = self.conn.execute(
            "SELECT * FROM models WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def promote(self, model_id):
        """将指定模型设为激活"""
        self.conn.execute("UPDATE models SET is_active = 0")
        self.conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        self.conn.commit()

    def list_all(self, limit=20):
        """列出所有模型，按 mAP50 降序"""
        rows = self.conn.execute(
            "SELECT id, name, version, mAP50, mAP50_95, precision, recall, "
            "attention_module, lightweight, onnx_path, is_active, created_at "
            "FROM models ORDER BY mAP50 DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, model_id):
        row = self.conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row else None

    def record_feedback_stats(self, date_str, total, fp, fn, used=0):
        """记录每日反馈统计"""
        existing = self.conn.execute(
            "SELECT id FROM feedback_stats WHERE date = ?", (date_str,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE feedback_stats SET total_received=?, false_positive=?, false_negative=?, used_for_training=? WHERE date=?",
                (total, fp, fn, used, date_str)
            )
        else:
            self.conn.execute(
                "INSERT INTO feedback_stats (date, total_received, false_positive, false_negative, used_for_training) "
                "VALUES (?, ?, ?, ?, ?)", (date_str, total, fp, fn, used)
            )
        self.conn.commit()

    def get_feedback_summary(self, days=30):
        rows = self.conn.execute(
            "SELECT date, total_received, false_positive, false_negative, used_for_training "
            "FROM feedback_stats ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
