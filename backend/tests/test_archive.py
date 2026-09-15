"""Archive API tests use tiny explicitly constructed fixtures; real archive is verified separately."""
import json
import sqlite3
from fastapi.testclient import TestClient

from app.archive.repository import ArchiveRepository
from app.main import app, get_archive


def fixture_archive(tmp_path):
    directory = tmp_path / 'archive'
    directory.mkdir()
    (directory / 'metadata.json').write_text(json.dumps({
        'source_name': 'Fixture', 'mode': 'historical_holdout_replay',
        'metrics': {'threshold_selected_on_validation': 0.6}}))
    db = sqlite3.connect(directory / 'replay.sqlite3')
    db.execute('CREATE TABLE replay_rows (row_id INTEGER PRIMARY KEY, time_offset_seconds REAL, amount REAL, actual_fraud INTEGER, fraud_score REAL)')
    db.executemany('INSERT INTO replay_rows VALUES (?, ?, ?, ?, ?)', [
        (10, 10, 12.5, 0, 0.1), (11, 11, 100, 1, 0.8), (12, 12, 15, 0, 0.3)])
    db.commit()
    db.close()
    return ArchiveRepository(directory)


def test_archive_cursor_order_and_retrospective_filter(tmp_path):
    archive = fixture_archive(tmp_path)
    first = archive.rows(after=-1, limit=2)
    assert [row['row_id'] for row in first['rows']] == [10, 11]
    assert first['has_more'] is True
    assert first['rows'][1]['model_flagged'] is True
    next_page = archive.rows(after=first['next_cursor'], limit=2)
    assert [row['row_id'] for row in next_page['rows']] == [12]
    assert [row['row_id'] for row in archive.rows(known_fraud_only=True)['rows']] == [11]


def test_archive_api_source_and_rows(tmp_path):
    archive = fixture_archive(tmp_path)
    app.dependency_overrides[get_archive] = lambda: archive
    try:
        with TestClient(app) as client:
            meta = client.get('/api/v1/archive/metadata')
            rows = client.get('/api/v1/archive/rows?after=-1&limit=2')
            bad = client.get('/api/v1/archive/rows?limit=101')
        assert meta.status_code == 200
        assert meta.json()['mode'] == 'historical_holdout_replay'
        assert rows.status_code == 200
        assert len(rows.json()['rows']) == 2
        assert bad.status_code == 422
    finally:
        app.dependency_overrides.clear()
