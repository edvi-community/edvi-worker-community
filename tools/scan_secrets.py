"""Check tracked source for new secrets without printing candidate values.

Install detect-secrets==1.5.0. The baseline contains reviewed synthetic fixtures and public hashes,
never raw secrets. New detections must be reviewed before updating the baseline.
"""
import json
import subprocess
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
with default_settings():
    found = SecretsCollection()
    for path in paths:
        if path and path != '.secrets.baseline' and (root / path).is_file():
            found.scan_file(str(root / path))
entries = []
for filename, values in found.data.items():
    for value in values:
        item = value.json()
        entries.append({'path': str(Path(filename).relative_to(root)),
                        'type': item['type'], 'hashed_secret': item['hashed_secret']})
known = json.loads((root / '.secrets.baseline').read_text())
new = [item for item in entries if item not in known]
for item in new:
    print(f"Unreviewed {item['type']} candidate in {item['path']} (value redacted)")
if new:
    raise SystemExit(1)
print(f'Checked tracked source; {len(entries)} detections match reviewed synthetic fixtures and public hashes.')
