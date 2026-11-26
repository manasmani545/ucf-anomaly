import json
from datetime import datetime
from pathlib import Path

root = Path('.')
docs_dir = root / 'docs'
docs_dir.mkdir(exist_ok=True)

log_path = docs_dir / 'experiment_log.md'

def append_entry(
    tag,
    status,
    epoch=None,
    train_loss=None,
    test_accuracy=None,
    checkpoint_path=None,
    notes=''
):
    timestamp = datetime.now().isoformat(timespec='seconds')

    entry = [
        f'## {timestamp} - {tag}',
        '',
        f'- Status: {status}',
    ]

    if epoch is not None:
        entry.append(f'- Epoch completed: {epoch}')

    if train_loss is not None:
        entry.append(f'- Train loss: {train_loss:.4f}')

    if test_accuracy is not None:
        entry.append(f'- Test accuracy: {test_accuracy:.4f}')

    if checkpoint_path:
        entry.append(f'- Checkpoint: {checkpoint_path}')

    if notes:
        entry.append(f'- Notes: {notes}')

    entry.extend(['', ''])

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(entry))

if __name__ == '__main__':
    append_entry(
        'account_a',
        'training_started',
        notes='UCF-Crime Top-K MIL baseline training started on local CPU.'
    )

    print(f'Documentation entry appended to {log_path}')
