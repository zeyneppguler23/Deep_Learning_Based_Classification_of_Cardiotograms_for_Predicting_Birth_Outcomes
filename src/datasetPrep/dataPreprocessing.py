'''
For the Multi-Classification task, this prepares the dataset by seperating evalution steps.
Evaluation steps include: 
1. 30 minutes long window beginning at maximum one hour before the end of the first stage of labor
2. 30 minutes long window beginning at maximum 30 minutes before the end of the first stage of labor
3. Full second stage of labor signal presented if five minutes or more of CTG signal was available
4. Evaluation of labor outcome – prediction of umbilical artery pH after delivery

Expert evaluation of the CTG data "Gold Standard" evaluation based on annotation of the signals by 9 expert obstetricians 
Note that this data is available at https://people.ciirc.cvut.cz/~spilkjir/data.html

Each expert obstetrician annotated the CTG signals as normal, suspicious, pathological or uninterpretable, which the dataset then be labelled using majority voting. 

Outputs:
    A dictionary with keys:
        step1, step2, step3, step4
    Each containing:
        (FHR_segment, UC_segment) or None if unavailable

Because labor stage boundary timestamps were not available,
we approximated the four evaluation windows using four consecutive 30-minute segments anchored to the end of recording (delivery): 
[-120,-90], [-90,-60], [-60,-30], and [-30,0] minutes

RUN PREPROCESSING: outputs = process_dataset(records_df, raw_dataset_path="../data/raw_dataset")
'''

import numpy as np
import wfdb 
from pathlib import Path


# =========================
# Signal Utilities
# =========================

def remove_trailing_zeros(signal):
    last_non_zero = len(signal) - 1
    while last_non_zero >= 0 and signal[last_non_zero] == 0:
        last_non_zero -= 1
    return signal[:last_non_zero + 1]


def downsample_to_1hz(signal, original_fs, target_fs=1):
    factor = int(original_fs // target_fs)
    return signal[::factor]


def is_signal_acceptable(signal, threshold=0.16):
    if not signal or len(signal) == 0:
        return False
    zero_count = signal.count(0)
    return (zero_count / len(signal)) < threshold


def pad_or_trim_edge(signal, target_length):
    """Edge-padding is better than constant-zero padding for CTG."""
    if len(signal) == target_length:
        return signal
    if len(signal) > target_length:
        return signal[:target_length]
    return np.pad(signal, (0, target_length - len(signal)), mode="edge")


# =========================
# Delivery-anchored windows
# =========================

def _extract_window_by_offset(fhr, uc, fs, end_idx, start_offset_sec, end_offset_sec):
    """
    Extract a window [end_idx - start_offset, end_idx - end_offset].
    Example: start_offset=120min, end_offset=90min  => [-120, -90] min.
    """
    start_idx = end_idx - int(start_offset_sec * fs)
    stop_idx  = end_idx - int(end_offset_sec * fs)

    if start_idx < 0 or stop_idx <= 0 or stop_idx <= start_idx:
        return None, None

    return fhr[start_idx:stop_idx], uc[start_idx:stop_idx]


def split_into_4_steps_delivery_anchored(fhr, uc, fs):
    """
    Steps are 30-min fixed windows relative to delivery (end of signal):
      step1: [-120, -90]
      step2: [-90,  -60]
      step3: [-60,  -30]
      step4: [-30,    0]
    """
    window_sec = 30 * 60
    end_idx = len(fhr)  # delivery = end of available signal

    # Offsets in seconds
    step1 = _extract_window_by_offset(fhr, uc, fs, end_idx, 4*window_sec, 3*window_sec)  # -120 to -90
    step2 = _extract_window_by_offset(fhr, uc, fs, end_idx, 3*window_sec, 2*window_sec)  # -90 to -60
    step3 = _extract_window_by_offset(fhr, uc, fs, end_idx, 2*window_sec, 1*window_sec)  # -60 to -30
    step4 = _extract_window_by_offset(fhr, uc, fs, end_idx, 1*window_sec, 0*window_sec)  # -30 to 0

    return {"step1": step1, "step2": step2, "step3": step3, "step4": step4}


# =========================
# Main: process one record
# =========================

def process_single_record(record_path, target_fs=1, apply_quality_filter=True, loss_threshold=0.16):
    """
    Load .hea/.dat record, clean, downsample, then split into 4 delivery-anchored windows.
    """
    record = wfdb.rdrecord(record_path)
    fs = record.fs

    signals = record.p_signal
    fhr = signals[:, 0].tolist()
    uc = signals[:, 1].tolist()

    # Clean trailing zeros (defines "delivery end" as last non-zero region)
    fhr = remove_trailing_zeros(fhr)
    uc = remove_trailing_zeros(uc)

    # Downsample to 1 Hz
    fhr = downsample_to_1hz(fhr, fs, target_fs)
    uc  = downsample_to_1hz(uc, fs, target_fs)
    fs = target_fs

    # Optional quality filtering
    if apply_quality_filter:
        if (not is_signal_acceptable(fhr, loss_threshold)) or (not is_signal_acceptable(uc, loss_threshold)):
            return None

    # Extract 4 steps relative to delivery
    steps = split_into_4_steps_delivery_anchored(fhr, uc, fs)

    # Enforce fixed-length 30-min windows -> 1800 samples at 1 Hz
    fixed_steps = {}
    for step, (f, u) in steps.items():
        if f is None or u is None:
            fixed_steps[step] = (None, None)
        else:
            fixed_steps[step] = (
                pad_or_trim_edge(np.array(f, dtype=np.float32), 1800),
                pad_or_trim_edge(np.array(u, dtype=np.float32), 1800),
            )

    return fixed_steps


# =========================
# Batch processing
# =========================

def process_dataset_from_folder(raw_dataset_path, record_ids=None, **kwargs):
    """
    Processes all records in a folder (WFDB .hea/.dat).
    If record_ids is None -> uses all .hea stems.

    Returns:
      outputs dict with:
        step1..step4: list of arrays (2,1800,1) or None
        valid_mask: per-step validity
        rec_ids: list of processed rec_ids
    """
    raw_dataset_path = Path(raw_dataset_path)

    if record_ids is None:
        record_ids = [p.stem for p in raw_dataset_path.glob("*.hea")]

    outputs = {
        "step1": [], "step2": [], "step3": [], "step4": [],
        "valid_mask": {"step1": [], "step2": [], "step3": [], "step4": []},
        "rec_ids": []
    }

    for rec_id in record_ids:
        rec_path = raw_dataset_path / rec_id
        steps = process_single_record(rec_path, **kwargs)

        if steps is None:
            continue

        outputs["rec_ids"].append(str(rec_id))

        for step in ["step1", "step2", "step3", "step4"]:
            fhr, uc = steps[step]
            valid = (fhr is not None) and (uc is not None)
            outputs["valid_mask"][step].append(valid)

            if valid:
                stacked = np.stack([fhr, uc], axis=0)[..., np.newaxis]  # (2,1800,1)
                outputs[step].append(stacked)
            else:
                outputs[step].append(None)

    return outputs
