import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.feature_dataset import FeatureDataset
from models.gru_attention_mil import GRUAttentionMIL


def append_documentation(
    tag,
    status,
    epoch=None,
    train_loss=None,
    test_accuracy=None,
    checkpoint_path=None,
    notes=''
):
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)

    log_path = docs_dir / 'experiment_log.md'

    entry = [
        f'## {datetime.now().isoformat(timespec="seconds")} - {tag}',
        '',
        f'- Status: {status}'
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cpu():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.backends.mkldnn.enabled = True

    os.environ.setdefault('OMP_NUM_THREADS', str(max(1, os.cpu_count() - 1)))
    os.environ.setdefault('MKL_NUM_THREADS', str(max(1, os.cpu_count() - 1)))
    os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'AVX512')

    print(f'CPU threads: {torch.get_num_threads()}')
    print(f'MKL enabled: {torch.backends.mkldnn.enabled}')


def build_loaders(config):
    train_dataset = FeatureDataset(
        config['dataset']['manifest'],
        'train',
        config['dataset']['num_classes'],
        config['dataset']['max_video_len']
    )

    test_dataset = FeatureDataset(
        config['dataset']['manifest'],
        'test',
        config['dataset']['num_classes'],
        config['dataset']['max_video_len']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()

    all_targets = []
    all_scores = []

    with torch.no_grad():
        for features, targets, masks, video_ids in tqdm(
            loader,
            desc='Evaluating',
            leave=False
        ):
            features = features.to(device)
            masks = masks.to(device)

            logits = model(features, masks)
            scores = torch.sigmoid(logits)

            all_targets.extend(targets[:, 1].cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    preds = (all_scores >= 0.5).astype(int)

    accuracy = float((preds == all_targets).mean())
    return accuracy, all_targets, all_scores


def save_checkpoint(path, model, optimizer, config, epoch, best_accuracy, history):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'history': history
    }, path)


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ucf_baseline.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--tag', default='ucf_baseline')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    configure_cpu()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = build_loaders(config)

    model = GRUAttentionMIL(
        feature_dim=config['dataset']['feature_dim'],
        num_classes=config['dataset']['num_classes'],
        k_ratio=config['model']['k_ratio']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_accuracy = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        history = checkpoint['history']

        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    append_documentation(
        args.tag,
        'training_started',
        notes='Feature normalization, gradient clipping, CPU optimization, and AVX-512-compatible MKL settings enabled.'
    )

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch:02d}/{config["training"]["epochs"]:02d}',
            leave=True
        )

        for features, targets, masks, video_ids in progress:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(features, masks)
            anomaly_targets = targets[:, 1]

            loss = criterion(logits, anomaly_targets)

            if not torch.isfinite(loss):
                progress.set_postfix(loss='NaN')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            progress.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = total_loss / max(total_samples, 1)

        test_accuracy, test_targets, test_scores = evaluate(model, test_loader, device)

        elapsed = time.time() - start_time

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_accuracy': test_accuracy,
            'seconds': elapsed
        })

        print(
            f'Epoch {epoch:02d} | '
            f'Train loss: {train_loss:.4f} | '
            f'Test accuracy: {test_accuracy:.4f} | '
            f'Time: {elapsed:.1f}s'
        )

        last_checkpoint = checkpoint_dir / f'last_{args.tag}.pt'

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            config,
            epoch,
            best_accuracy,
            history
        )

        append_documentation(
            args.tag,
            'epoch_completed',
            epoch=epoch,
            train_loss=train_loss,
            test_accuracy=test_accuracy,
            checkpoint_path=str(last_checkpoint),
            notes='Per-video feature normalization and gradient clipping applied.'
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            best_checkpoint = checkpoint_dir / f'best_{args.tag}.pt'

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                config,
                epoch,
                best_accuracy,
                history
            )

            append_documentation(
                args.tag,
                'best_checkpoint_saved',
                epoch=epoch,
                train_loss=train_loss,
                test_accuracy=test_accuracy,
                checkpoint_path=str(best_checkpoint),
                notes='New best test accuracy achieved.'
            )

    with open(log_dir / f'history_{args.tag}.json', 'w') as f:
        json.dump(history, f, indent=2)

    append_documentation(
        args.tag,
        'training_completed',
        notes=f'Best test accuracy: {best_accuracy:.4f}'
    )

    print(f'Best test accuracy: {best_accuracy:.4f}')


if __name__ == '__main__':
    main()
