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
'''

import numpy as np
import wfdb 
from pathlib import Path

# Signal Utilities / Functions 
def remove_trailing_zeros(signal):
    """Remove trailing zeros from a signal."""
    last_non_zero = len(signal) - 1
    while last_non_zero >= 0 and signal[last_non_zero] == 0:
        last_non_zero -= 1 
    return signal[:last_non_zero + 1]


def downsample_to_1hz(signal, original_fs, target_fs=1):
    """Downsample a signal to a target frequency."""
    factor = int(original_fs // target_fs)
    return signal[::factor]

def sec_to_idx(seconds, fs):
    """Convert seconds to sample index based on sampling frequency."""
    return int(round(seconds * fs))

def pad_or_trim(signal, target_length):
    """Pad with zeros or trim the signal to match the target length."""
    if len(signal) == target_length:
        return signal
    elif len(signal) > target_length:
        return signal[:target_length]
    else:
        return np.pad(signal, (0, target_length - len(signal)), mode='constant')
    

#Clinical window extraction functions

def extract_step1(fhr, uc, fs, end_first_stage_sec):
    """30-min window ending ≤60 min before end of 1st stage."""
    window_sec = 30 * 60
    max_offset_sec = 60 * 60

    end_idx = sec_to_idx(end_first_stage_sec - max_offset_sec, fs)
    start_idx = end_idx - sec_to_idx(window_sec, fs)

    if start_idx < 0 or end_idx <= 0:
        return None, None

    return fhr[start_idx:end_idx], uc[start_idx:end_idx]

def extract_step2(fhr, uc, fs, end_first_stage_sec):
    """30-min window ending ≤30 min before end of 1st stage."""
    window_sec = 30 * 60
    max_offset_sec = 30 * 60

    end_idx = sec_to_idx(end_first_stage_sec - max_offset_sec, fs)
    start_idx = end_idx - sec_to_idx(window_sec, fs)

    if start_idx < 0 or end_idx <= 0:
        return None, None

    return fhr[start_idx:end_idx], uc[start_idx:end_idx]

def extract_step3(fhr, uc, fs, start_second_stage_sec, delivery_sec):
    """Full second stage if ≥5 min of signal available."""
    start_idx = sec_to_idx(start_second_stage_sec, fs)
    end_idx = sec_to_idx(delivery_sec, fs)

    if end_idx <= start_idx:
        return None, None

    if (end_idx - start_idx) < sec_to_idx(5 * 60, fs):
        return None, None

    return fhr[start_idx:end_idx], uc[start_idx:end_idx]

def extract_step4(fhr, uc, fs, delivery_sec):
    """30-min window ending at delivery."""
    window_sec = 30 * 60

    end_idx = sec_to_idx(delivery_sec, fs)
    start_idx = end_idx - sec_to_idx(window_sec, fs)

    if start_idx < 0 or end_idx <= 0:
        return None, None

    return fhr[start_idx:end_idx], uc[start_idx:end_idx]


# Main function to process a single record
def process_single_record(
    record_path,
    end_first_stage_sec,
    start_second_stage_sec,
    delivery_sec,
    target_fs = 1
):
    """
    load .hea/.dat record, clean, downsample amd extract 4 clinical windows
    """

    record =wfdb.rdrecord(record_path)
    fs = record.fs

    signals = record.p_signal
    fhr =signals[:, 0].tolist()
    uc = signals[:, 1].tolist()

    #cleaning
    fhr = remove_trailing_zeros(fhr)
    uc = remove_trailing_zeros(uc)

    #downsampling 
    fhr = downsample_to_1hz(fhr, fs, target_fs)
    uc = downsample_to_1hz(uc, fs, target_fs)
    fs = target_fs

    #extracting clinical windows
    s1 = extract_step1(fhr, uc, fs, end_first_stage_sec)
    s2 = extract_step2(fhr, uc, fs, end_first_stage_sec)
    s3 = extract_step3(fhr, uc, fs, start_second_stage_sec, delivery_sec)
    s4 = extract_step4(fhr, uc, fs, delivery_sec)

    return {
        "step1": s1,
        "step2": s2,
        "step3": s3,
        "step4": s4
    }

# Batch dataset processing function

def process_dataset(records_df, raw_dataset_path):
        """
        records_df must contain:
            rec_id
            end_first_stage_sec
            start_second_stage_sec
            delivery_sec

        Returns:
            dict with step1–step4 arrays shaped (N, 2, T, 1)
        """
        
        raw_dataset_path = Path(raw_dataset_path)

        outputs = {
            "step1": [],
            "step2": [],
            "step3": [],
            "step4": [],
            "valid_mask": {
                "step1": [],
                "step2": [],
                "step3": [],
                "step4": []
            },
            "rec_ids": []
        }

        for _, row in records_df.iterrows():
            rec_id = row["rec_id"]
            rec_path = raw_dataset_path / rec_id

            windows = process_single_record(
                rec_path,
                end_first_stage_sec=row["end_first_stage_sec"],
                start_second_stage_sec=row["start_second_stage_sec"],
                delivery_sec=row["delivery_sec"]
            )

            if windows is None:
                continue

            outputs["rec_ids"].append(rec_id)

            for step in ["step1", "step2", "step3", "step4"]:
                fhr, uc = windows[step]
                valid = (fhr is not None) and (uc is not None)

                outputs["valid_mask"][step].append(valid)

                if valid:
                    # Fixed 30-min windows → pad/trim to 1800 samples
                    if step in ["step1", "step2", "step4"]:
                        fhr = pad_or_trim(fhr, 1800)
                        uc = pad_or_trim(uc, 1800)

                    stacked = np.stack([fhr, uc], axis=0)[..., np.newaxis]
                    outputs[step].append(stacked)
                else:
                    outputs[step].append(None)

        return outputs