"""
Filename: plot_timechan.py
Author: Siyang Li
Date: 2026-01-23
Description: Time-channel plot of converting EEG to stacked waveform images.
             Requires the following files:
             1. EEG trials, in the form of numpy array, of shape (num_trials, num_channels, num_samples), at root_path/X.npy
             2. Corresponding labels, in the form of numpy array, of shape (num_trials, ), of 0 meaning non-seizure and 1 meaning seizure, at root_path/y.npy
             Then read the parser of config arguments.
             Two sets of plots will be created:
             1. VLM, which has axis, ticks, channel names as text.
             2. Vision, which does not have those, only having EEG waveform.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import time


class EEGProcessor:
    def __init__(self, output_dir):
        self.root_dir = output_dir
        self.sim_dir = os.path.join(output_dir, "simulation_checks")
        self.vision_root = os.path.join(output_dir, "Vision")
        self.vlm_root = os.path.join(output_dir, "VLM")
        self.colors = plt.get_cmap('tab20')

        # Create directories
        os.makedirs(self.sim_dir, exist_ok=True)
        os.makedirs(self.vision_root, exist_ok=True)
        os.makedirs(self.vlm_root, exist_ok=True)

    @staticmethod
    def _robust_norm_and_clip(signal, spacing=1.0, safety_margin=0.95):
        low, high = np.percentile(signal, [1, 99])
        denom = (high - low) if (high - low) != 0 else 1.0
        norm = ((signal - low) / denom) - 0.5
        limit = 0.5 * safety_margin
        # in-place clip if possible, but numpy creates copy usually
        clipped = np.clip(norm, -limit, limit)
        return clipped * spacing

    @staticmethod
    def _get_save_path(root_path, subject_id, class_label, trial_idx, suffix):
        sub_dir = os.path.join(root_path, f"sub_{subject_id:02d}")
        class_dir = os.path.join(sub_dir, f"class_{int(class_label)}")
        os.makedirs(class_dir, exist_ok=True)
        filename = f"trial_{trial_idx}_{suffix}.png"
        return os.path.join(class_dir, filename)

    @staticmethod
    def plot_vision(ax, trial_data, spacing, colors):
        ax.clear()
        num_channels, num_samples = trial_data.shape

        for i in range(num_channels):
            sig = EEGProcessor._robust_norm_and_clip(trial_data[i], spacing=spacing, safety_margin=0.95)  # safety_margin is whether allowing overlapping of channels. if set to lower than 1, no overlapping happens but high/low amplitude values get cut off.
            y_pos = (num_channels - 1 - i) * spacing
            ax.plot(sig + y_pos, color=colors(i), linewidth=3.0)

        ax.set_xlim(0, num_samples)
        ax.set_ylim(-0.5 * spacing, (num_channels - 0.5) * spacing)
        ax.axis('off')

    @staticmethod
    def plot_vlm(ax, trial_data, sfreq, spacing, colors, ch_names):
        ax.clear()
        num_channels, num_samples = trial_data.shape
        duration = num_samples / sfreq
        time_vec = np.arange(num_samples) / sfreq

        for i in range(num_channels):
            sig = EEGProcessor._robust_norm_and_clip(trial_data[i], spacing=spacing, safety_margin=0.95)
            y_pos = (num_channels - 1 - i) * spacing
            ax.plot(time_vec, sig + y_pos, color=colors(i), linewidth=2.0)

        ax.set_yticks(np.arange(num_channels) * spacing)
        ax.set_yticklabels(ch_names[::-1], fontsize=16, fontweight='heavy', family='sans-serif')

        tick_interval = 0.5
        ticks = np.arange(0, duration + tick_interval, tick_interval)
        ax.set_xticks(ticks)
        ax.set_xlabel("Time (s)", fontsize=18, fontweight='heavy')
        ax.tick_params(axis='x', labelsize=16, width=2)
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5 * spacing, (num_channels - 0.5) * spacing)
        ax.grid(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)

    def run_preprocessing_simulations(self, vision_path, vlm_path):
        """ Run checks on Trial 0 images only """
        print(f"Running Preprocessing Simulations...")
        try:
            img = Image.open(vision_path).convert('RGB')
            img_resized = img.resize((256, 256), Image.BICUBIC)
            left, top = (256 - 224) / 2, (256 - 224) / 2
            right, bottom = (256 + 224) / 2, (256 + 224) / 2
            img_cropped = img_resized.crop((left, top, right, bottom))

            save_path = os.path.join(self.sim_dir, "Trial0_Sim_ResNet_ViT_Input(224x224).jpg")
            img_cropped.save(save_path, quality=95)
            print(f"  [Vision Sim] Saved: {save_path}")
        except Exception as e:
            print(f"  [Vision Sim] Error: {e}")

        try:
            img = Image.open(vlm_path).convert('RGB')
            save_path = os.path.join(self.sim_dir, "Trial0_Sim_API_Compression(Quality50).jpg")
            img.save(save_path, "JPEG", quality=50)
            print(f"  [API Sim]    Saved: {save_path}")
        except Exception as e:
            print(f"  [API Sim]    Error: {e}")


# ==========================================
# WORKER FUNCTION
# ==========================================
def process_batch(indices, x_path, y_path, subject_map,
                  output_dir, ch_names, sfreq):

    X_mmap = np.load(x_path, mmap_mode='r')
    y_full = np.load(y_path)

    processor_ref = EEGProcessor(output_dir)

    # Vision Figure
    fig_vis, ax_vis = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    fig_vis.patch.set_facecolor('white')

    # VLM Figure
    fig_vlm, ax_vlm = plt.subplots(figsize=(10.24, 10.24), dpi=100)
    fig_vlm.patch.set_facecolor('white')

    num_channels = len(ch_names)
    cmap = plt.get_cmap('tab20')
    colors = cmap.resampled(num_channels)
    spacing = 1.0

    processed_count = 0

    try:
        for idx in indices:
            # read data
            trial_data = X_mmap[idx]
            label = y_full[idx]
            sub_id = subject_map[idx]

            # --- Generate Vision ---
            EEGProcessor.plot_vision(ax_vis, trial_data, spacing, colors)
            plt.tight_layout(pad=0)
            save_path_vis = processor_ref._get_save_path(
                processor_ref.vision_root, sub_id, label, idx, "Vision"
            )
            fig_vis.savefig(save_path_vis, bbox_inches='tight', pad_inches=0)

            # --- Generate VLM ---
            EEGProcessor.plot_vlm(ax_vlm, trial_data, sfreq, spacing, colors, ch_names)
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
            save_path_vlm = processor_ref._get_save_path(
                processor_ref.vlm_root, sub_id, label, idx, "VLM"
            )
            fig_vlm.savefig(save_path_vlm, bbox_inches='tight', pad_inches=0.1)

            processed_count += 1

            if processed_count % 100 == 0:
                ax_vis.clear()
                ax_vlm.clear()

    except Exception as e:
        print(f"Worker Error on index {idx}: {e}")
    finally:
        plt.close(fig_vis)
        plt.close(fig_vlm)

    return processed_count


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Configuration
    root_path = '/'
    data_name = 'CHSZ'
    # data_name = 'NICU'

    if data_name == 'CHSZ':
        input_file_X = root_path + "CHSZ/X.npy"
        input_file_y = root_path + "CHSZ/y.npy"
        output_dir = root_path + "CHSZ/time-chan/"
        sfreq = 250  # sampling frequency in Hz
        subject_counts = [191,41,121,62,62,144,26,37,66,3599,2738,3599,254,2569,185,42,97,612,1337,1047,105,66,85,97,118,328,3599] # count of each subject's trial number
        channels_order = [
            "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
            "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz"
        ]
    elif data_name == 'NICU':
        input_file_X = root_path + "NICU/X.npy"
        input_file_y = root_path + "NICU/y.npy"
        output_dir = root_path + "NICU/time-chan/"
        sfreq = 256
        subject_counts = [1747,856,960,913,887,1871,3853,931,1724,1484,1373,2251,996,1426,960,1677,882,1622,1269,1523,1157,1458,2420,839,901,2462,1175,979,1462,974,2837,1224,992,932,988,954,1048,1242,824]
        channels_order = [
            "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
            "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz"
        ]

    print(f"Initializing...")

    # 2. Prepare Directories & First Check
    processor = EEGProcessor(output_dir)

    # Load Y solely to get total count (X is loaded via mmap inside workers)
    # Load X in mmap mode just to get shape and run simulation once
    X_mmap = np.load(input_file_X, mmap_mode='r')
    y_full = np.load(input_file_y)
    total_trials = X_mmap.shape[0]

    print(f"Data Shape: {X_mmap.shape}, Labels: {y_full.shape}")

    # Check alignment
    if sum(subject_counts) != total_trials:
        print(f"WARNING: Subject counts sum ({sum(subject_counts)}) != total trials ({total_trials}).")

    # 3. Create Subject Mapping Array
    print("Generating subject mapping...")
    trial_to_subject = np.zeros(total_trials, dtype=int)
    current_idx = 0
    for sub_id, count in enumerate(subject_counts):
        trial_to_subject[current_idx: current_idx + count] = sub_id
        current_idx += count

    # 4. Run Simulation on Trial 0 (Main Process)
    print("Running initial simulation on Trial 0...")
    # Generate trial 0 locally for simulation check
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    processor.plot_vision(ax, X_mmap[0], 1.0, processor.colors.resampled(len(channels_order)))
    plt.tight_layout(pad=0)
    sim_vis_path = processor._get_save_path(processor.vision_root, 0, y_full[0], 0, "Vision")
    plt.savefig(sim_vis_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)
    processor.plot_vlm(ax, X_mmap[0], sfreq, 1.0, processor.colors.resampled(len(channels_order)), channels_order)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    sim_vlm_path = processor._get_save_path(processor.vlm_root, 0, y_full[0], 0, "VLM")
    plt.savefig(sim_vlm_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    processor.run_preprocessing_simulations(sim_vis_path, sim_vlm_path)

    # 5. Multiprocessing Execution
    # Determine chunk size. For 20k images, splitting into logical chunks helps tracking.
    # Using all available cores.
    num_workers = min(os.cpu_count(), 32)  # Cap at 32 to avoid IO thrashing if on huge server
    print(f"Starting parallel processing with {num_workers} workers...")

    # Split indices into chunks
    indices = np.arange(total_trials)
    # Exclude trial 0 if you don't want to overwrite (or just overwrite it, it's fine)
    # indices = indices[1:]

    chunks = np.array_split(indices, num_workers * 4)  # 4 tasks per worker generally balances load well

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare function arguments
        # We pass file paths instead of big arrays
        futures = []
        for chunk_indices in chunks:
            if len(chunk_indices) == 0: continue
            futures.append(
                executor.submit(
                    process_batch,
                    chunk_indices,
                    input_file_X,
                    input_file_y,
                    trial_to_subject,
                    output_dir,
                    channels_order,
                    sfreq
                )
            )

        # Monitor progress
        total_processed = 0
        for i, f in enumerate(futures):
            count = f.result()  # This blocks until the specific chunk is done
            total_processed += count
            elapsed = time.time() - t0
            rate = total_processed / elapsed
            print(f"  [{total_processed}/{total_trials}] Processed. Rate: {rate:.2f} trials/sec")

    print(f"\nProcessing Complete. Total time: {time.time() - t0:.2f}s")