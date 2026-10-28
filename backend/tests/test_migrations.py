"""The pre-Alembic asl.db (create_all schema + synthetic rows) must upgrade without data loss."""
import sqlite3

from sqlalchemy import create_engine, inspect

from app.db import migrate


def test_legacy_database_upgrades(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE signs (id INTEGER PRIMARY KEY, label VARCHAR NOT NULL, frames JSON NOT NULL,
                            video_url VARCHAR, source VARCHAR);
        CREATE INDEX ix_signs_label ON signs (label);
        CREATE TABLE attempts (id INTEGER PRIMARY KEY, sign_id INTEGER, score FLOAT,
                               created_at DATETIME DEFAULT (CURRENT_TIMESTAMP));
        CREATE INDEX ix_attempts_sign_id ON attempts (sign_id);
        INSERT INTO signs (label, frames, source) VALUES ('A', '[[0]]', 'alphabet'), ('J', '[[0]]', 'alphabet'),
            ('thank you', '[[0]]', 'common'), ('coffee', '[[0]]', 'user');
        INSERT INTO attempts (sign_id, score) VALUES (1, 50.0);
    """)
    con.commit()
    con.close()

    url = f"sqlite:///{path.as_posix()}"
    migrate(url)
    eng = create_engine(url)
    insp = inspect(eng)
    assert {"reference_clips", "reference_videos", "learners", "alembic_version"} <= set(insp.get_table_names())
    with eng.connect() as c:
        rows = {r[0]: r[1:] for r in c.exec_driver_sql(
            "SELECT label, category, kind, frames_source, gloss FROM signs")}
        assert rows["A"] == ("alphabet", "static", "synthetic", "A")
        assert rows["J"] == ("alphabet", "dynamic", "synthetic", "J")
        assert rows["thank you"] == ("word", "dynamic", "synthetic", "THANK-YOU")
        assert rows["coffee"][0] == "user" and rows["coffee"][2] == "legacy_user"
        assert c.exec_driver_sql("SELECT COUNT(*) FROM attempts").scalar() == 1
        # frames is now nullable: vocabulary entries without synthetic data are allowed
        c.exec_driver_sql("INSERT INTO signs (label, category) VALUES ('water', 'word')")
    eng.dispose()
    migrate(url)  # idempotent


def test_fresh_database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    migrate(url)
    eng = create_engine(url)
    cols = {c["name"] for c in inspect(eng).get_columns("attempts")}
    assert {"learner_id", "components", "hints", "model_version"} <= cols
    eng.dispose()
