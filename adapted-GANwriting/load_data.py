"""
load_data.py — DohaScript (Devanagari) adaptation of GANwriting data loader.

Groundtruth format (groundtruth.txt):
    images/WRITER_line_XX_word_YY.png<TAB>word

Directory layout expected:
    DATASET_ROOT/
        images/              ← word-level PNG crops
        groundtruth.txt      ← as produced by your segmentation pipeline
"""

import os
import random
import cv2
import numpy as np
import torch.utils.data as D

# ─── User config ─────────────────────────────────────────────────────────────
DATASET_ROOT = '/mnt/xdrive/guest/Laurel_Hardy/word_seg/word_dataset'          # root folder containing images/ and groundtruth.txt
GT_FILE      = os.path.join(DATASET_ROOT, 'groundtruth.txt')
TRAIN_RATIO  = 0.85
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────────────────────────────

IMG_HEIGHT    = 64
IMG_WIDTH     = 256          # wider than original (216) to fit Devanagari
MAX_CHARS     = 12           # longest word in dataset is 10 codepoints
NUM_CHANNEL   = 50
EXTRA_CHANNEL = NUM_CHANNEL + 1
OUTPUT_MAX_LEN = MAX_CHARS + 2   # GO + word + END

tokens    = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary & writer mapping  (built once at import time)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_groundtruth(gt_file):
    """Return list of (img_relative_path, writer_id, word)."""
    records = []
    with open(gt_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            img_path, word = parts[0], parts[1]
            # filename pattern: WRITER_line_XX_word_YY.png
            basename = os.path.basename(img_path)
            writer   = basename.split('_line_')[0]
            records.append((img_path, writer, word))
    return records


_all_records = _parse_groundtruth(GT_FILE)

# Character vocabulary — built from every unicode codepoint seen in the corpus
_all_chars   = sorted(set(ch for _, _, w in _all_records for ch in w))
letter2index = {ch: i for i, ch in enumerate(_all_chars)}
index2letter = {i: ch for i, ch in enumerate(_all_chars)}
num_classes  = len(_all_chars)
vocab_size   = num_classes + num_tokens

# Writer mapping
_all_writers = sorted(set(w for _, w, _ in _all_records))
NUM_WRITERS  = len(_all_writers)
wid2label    = {w: i for i, w in enumerate(_all_writers)}

# OOV word corpus — all unique words from the full dataset
text_corpus  = list(set(word for _, _, word in _all_records))


# ═══════════════════════════════════════════════════════════════════════════════
# Character-level edit for Devanagari (used only when OOV=False)
# ═══════════════════════════════════════════════════════════════════════════════

def edits1(word, min_len=1, max_len=MAX_CHARS):
    """One random character-level edit using the Devanagari vocabulary."""
    letters = _all_chars
    chars = list(word)
    n = len(chars)
    ops = []
    if n > min_len:
        # delete
        i = random.randrange(n)
        ops.append(''.join(chars[:i] + chars[i+1:]))
    if n < max_len:
        # insert
        i = random.randrange(n + 1)
        ops.append(''.join(chars[:i] + [random.choice(letters)] + chars[i:]))
    # substitute
    i = random.randrange(n) if n else 0
    ops.append(''.join(chars[:i] + [random.choice(letters)] + chars[i+1:]))
    # transpose
    if n > 1:
        i = random.randrange(n - 1)
        t = chars[:]
        t[i], t[i+1] = t[i+1], t[i]
        ops.append(''.join(t))
    result = random.choice(ops) if ops else word
    return result[:max_len] if result else word[:1]


# ═══════════════════════════════════════════════════════════════════════════════
# Train / test split by writer
# ═══════════════════════════════════════════════════════════════════════════════

def _split_writers(all_writers, ratio, seed):
    rng = random.Random(seed)
    writers = all_writers[:]
    rng.shuffle(writers)
    cut = int(len(writers) * ratio)
    return set(writers[:cut]), set(writers[cut:])


_train_writers, _test_writers = _split_writers(_all_writers, TRAIN_RATIO, RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class DohaWords(D.Dataset):
    """
    Each __getitem__ returns data for one writer:
        (domain, wid_int, idx_list, imgs, img_widths, labels,
         img_xt, label_xt, label_xt_swap)
    matching the original GANwriting tuple structure.
    """

    def __init__(self, data_dict, oov):
        """
        data_dict: {wid_int: [(img_path, word_str), ...]}
        """
        self.data_dict    = data_dict
        self.oov          = oov
        self.output_max_len = OUTPUT_MAX_LEN
        # DataLoader passes positional indices 0..N-1; map them to actual keys
        self._keys = sorted(data_dict.keys())

    # ── helpers ──────────────────────────────────────────────────────────────

    def _safe_word(self, word):
        """Keep only characters in our vocabulary; truncate to MAX_CHARS."""
        filtered = ''.join(ch for ch in word if ch in letter2index)
        return filtered[:MAX_CHARS] if filtered else list(letter2index.keys())[0]

    def _label_padding(self, word):
        word = self._safe_word(word)
        ll = [letter2index[ch] + num_tokens for ch in word]
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        pad = self.output_max_len - len(ll)
        if pad > 0:
            ll += [tokens['PAD_TOKEN']] * pad
        elif pad < 0:
            # truncate (shouldn't happen if MAX_CHARS is set correctly)
            ll = ll[:self.output_max_len - 1] + [tokens['END_TOKEN']]
        return ll

    def _new_ed1(self, label_arr):
        """Decode label → word, apply one edit, re-encode."""
        arr = list(label_arr)
        try:
            start = arr.index(tokens['GO_TOKEN'])
            end   = arr.index(tokens['END_TOKEN'])
        except ValueError:
            return np.array(self._label_padding(random.choice(text_corpus)))
        word = ''.join(index2letter[i - num_tokens]
                       for i in arr[start+1:end]
                       if i - num_tokens in index2letter)
        new_word = edits1(word) if word else random.choice(text_corpus)
        return np.array(self._label_padding(new_word))

    def read_image(self, img_path):
        """Read and normalise a word image to (IMG_HEIGHT, IMG_WIDTH)."""
        url = os.path.join(DATASET_ROOT, img_path)
        img = cv2.imread(url, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32'), 0

        rate  = IMG_HEIGHT / img.shape[0]
        img   = cv2.resize(img,
                           (max(1, int(img.shape[1] * rate)), IMG_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)
        img   = img / 255.0
        img   = 1.0 - img          # invert: text=1, background=0
        w     = img.shape[1]
        img_w = min(w, IMG_WIDTH)

        canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        canvas[:, :img_w] = img[:, :img_w]

        mean, std = 0.5, 0.5
        canvas = (canvas - mean) / std
        return canvas.astype('float32'), img_w

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, pos_idx):
        wid_int = self._keys[pos_idx]   # translate positional → writer key
        samples = self.data_dict[wid_int]
        np.random.shuffle(samples)

        imgs, img_widths, labels, idxs = [], [], [], []
        for img_path, word in samples:
            img, img_w = self.read_image(img_path)
            label      = self._label_padding(word)
            imgs.append(img)
            img_widths.append(img_w)
            labels.append(label)
            idxs.append(img_path)

        # Pad / tile to EXTRA_CHANNEL
        n = len(imgs)
        if n >= EXTRA_CHANNEL:
            imgs       = imgs[:EXTRA_CHANNEL]
            img_widths = img_widths[:EXTRA_CHANNEL]
            labels     = labels[:EXTRA_CHANNEL]
            idxs       = idxs[:EXTRA_CHANNEL]
        else:
            while len(imgs) < EXTRA_CHANNEL:
                need = EXTRA_CHANNEL - len(imgs)
                imgs       = imgs       + imgs[:need]
                img_widths = img_widths + img_widths[:need]
                labels     = labels     + labels[:need]
                idxs       = idxs       + idxs[:need]

        imgs_arr = np.stack(imgs, axis=0)   # (EXTRA_CHANNEL, H, W)

        # Pick one sample as the "target"
        _id     = np.random.randint(EXTRA_CHANNEL)
        img_xt  = imgs_arr[_id:_id+1]       # (1, H, W)

        if self.oov:
            # OOV: random words from corpus (core training objective)
            label_xt      = np.array(self._label_padding(
                                self._safe_word(random.choice(text_corpus))))
            label_xt_swap = np.array(self._label_padding(
                                self._safe_word(random.choice(text_corpus))))
        else:
            label_xt      = np.array(labels[_id])
            label_xt_swap = self._new_ed1(labels[_id])

        imgs_arr   = np.delete(imgs_arr,   _id, axis=0)  # (NUM_CHANNEL, H, W)
        img_widths = np.delete(np.array(img_widths, dtype='int64'), _id)
        labels     = np.delete(np.array(labels,     dtype='int64'), _id, axis=0)
        idxs       = [idxs[i] for i in range(len(idxs)) if i != _id]

        return (
            'src',
            wid_int,
            idxs,
            imgs_arr,
            img_widths,
            labels,
            img_xt,
            label_xt,
            label_xt_swap,
        )

    def __len__(self):
        return len(self.data_dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Public loader
# ═══════════════════════════════════════════════════════════════════════════════

def loadData(oov):
    """Returns (train_dataset, test_dataset) as DohaWords instances."""
    tr_dict, te_dict = {}, {}

    # Group records by writer integer label
    for img_path, writer, word in _all_records:
        wid = wid2label[writer]
        entry = (img_path, word)
        if writer in _train_writers:
            tr_dict.setdefault(wid, []).append(entry)
        else:
            te_dict.setdefault(wid, []).append(entry)

    data_train = DohaWords(tr_dict, oov)
    data_test  = DohaWords(te_dict, oov)

    print(f'[DohaScript] vocab={vocab_size} chars={num_classes}  '
          f'writers total={NUM_WRITERS} '
          f'(train={len(tr_dict)} test={len(te_dict)})')
    return data_train, data_test


if __name__ == '__main__':
    tr, te = loadData(oov=True)
    print(f'Train samples: {len(tr)}, Test samples: {len(te)}')
    sample = tr[0]
    print('imgs shape:', sample[3].shape)
    print('label_xt shape:', sample[7].shape)