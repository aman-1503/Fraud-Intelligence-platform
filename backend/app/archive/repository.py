"""Read-only access to a time-ordered held-out real-world archive."""
import json
import sqlite3
from pathlib import Path


class ArchiveRepository:
    def __init__(self, directory):
        self.directory = Path(directory)

    def available(self):
        return (self.directory / 'metadata.json').is_file() and (self.directory / 'replay.sqlite3').is_file()

    def metadata(self):
        return json.loads((self.directory / 'metadata.json').read_text())

    def rows(self, after=-1, limit=20, known_fraud_only=False):
        if not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        if after < -1:
            raise ValueError('after must be -1 or a row ID')
        db = sqlite3.connect(f'file:{self.directory / "replay.sqlite3"}?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            query = 'SELECT row_id, time_offset_seconds, amount, actual_fraud, fraud_score FROM replay_rows WHERE row_id > ?'
            if known_fraud_only:
                query += ' AND actual_fraud = 1'
            query += ' ORDER BY row_id LIMIT ?'
            found = [dict(row) for row in db.execute(query, (after, limit + 1))]
        finally:
            db.close()
        more = len(found) > limit
        rows = found[:limit]
        threshold = self.metadata()['metrics']['threshold_selected_on_validation']
        for row in rows:
            row['model_flagged'] = row['fraud_score'] >= threshold
        return {'rows': rows, 'next_cursor': rows[-1]['row_id'] if rows else after,
                'has_more': more, 'mode': 'historical_holdout_replay'}
