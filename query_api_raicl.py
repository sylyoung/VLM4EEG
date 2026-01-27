"""
Filename: query_api_raicl.py
Author: Siyang Li
Date: 2026-01-23
Description: Query API VLM (Gemini in this code) and using RAICL strategies to select few-shot examples.
             Requires your own seizure EEG data (detailed description in plot_timechan.py) and running these files first:
             1. plot_timechan.py
             2. extract_feature.py
             This file requires the following files:
             1. plotted EEG waveform images, by plot_timechan.py
             2. features of the waveform images, extracted by visual encoder (e.g. CLIP visual encoder), by extract_feature.py
             3. prompt file of prompt-Z.txt
             4. Gemini API key
             Please read the parser of config arguments.
"""
import sys, os, argparse, re, asyncio, csv, warnings, pickle, glob
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_distances
from tqdm.asyncio import tqdm_asyncio
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

warnings.filterwarnings("ignore")
DEBUG_MODE = True


# ==========================================
# 0. ENV & SETUP
# ==========================================
def setup_env():
    proxy_url = "http://xxx.xxx.xxx.xxx:xxx"
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['GRPC_PROXY_EXP'] = proxy_url
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
    os.environ['GRPC_POLL_STRATEGY'] = 'poll'
    print(f"DEBUG: Proxy set to {proxy_url}")


setup_env()


def load_raw_index(dataset):
    path = Path(f"./index_{dataset}_RawVectors.pkl")
    if not path.exists():
        sys.exit(f"CRITICAL: Index {path} not found. Run build_raw_index.py.")
    with open(path, 'rb') as f: return pickle.load(f)


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==========================================
# 1. SELECTION ALGORITHMS
# ==========================================
def get_medoids_with_distance(pool, target_center, k):
    if not pool or k == 0: return []

    # Extract vectors
    pool_vecs = np.stack([x['vec'] for x in pool])
    dists = cosine_distances(pool_vecs, target_center.reshape(1, -1)).flatten()
    sorted_indices = np.argsort(dists)

    # Select best K (unique)
    selected = []
    for i in sorted_indices[:len(pool)]:
        item = pool[i]
        # Copy item to avoid mutating index
        result_item = {'path': item['path'], 'label': item['label'], 'dist': float(dists[i])}
        selected.append(result_item)
        if len(selected) == k: break

    # PAD WITH DUPLICATES IF NEEDED
    if len(selected) < k and len(selected) > 0:
        while len(selected) < k:
            selected.append(selected[0].copy())

    return selected


def select_examples(index, subject, test_path, mode, k, nearest_neighbor=False):
    """
    Selection Logic with Padding.
    - NS: Representative (Medoid of self history).
    - SZ: Nearest Medoid (if NN=True) or Representative (if NN=False).
    """
    m = re.search(r'trial_(\d+)', test_path)
    curr_trial = int(m.group(1)) if m else 999999

    # --- 1. NON-SEIZURE POOL ---
    self_ns_pool = []
    if subject in index:
        # Strict filter for class_0
        self_ns_pool = [t for t in index[subject]
                        if t['label'] == 0 and "class_0" in t['path']
                        and t['trial'] < curr_trial and t['trial'] != -1]

        if len(self_ns_pool) < k:
            self_ns_pool.extend([t for t in index[subject]
                                 if t['label'] == 0 and "class_0" in t['path']
                                 and t['trial'] > curr_trial])

    other_subjects = [s for s in index.keys() if s != subject]

    sel_ns = []
    sel_sz = []

    # --- A. NON-SEIZURE SELECTION ---
    if self_ns_pool:
        vecs = np.stack([x['vec'] for x in self_ns_pool])
        centroid = np.mean(vecs, axis=0)
        sel_ns = get_medoids_with_distance(self_ns_pool, centroid, k)
        for x in sel_ns: x['type'] = 'RepNS'

    # --- B. SEIZURE SELECTION ---
    if nearest_neighbor:
        medoid_pool = []
        for sub in other_subjects:
            # Strict filter for class_1
            sub_trials = [t for t in index[sub] if t['label'] == 1 and "class_1" in t['path']]
            if not sub_trials: continue

            s_mean = np.mean(np.stack([t['vec'] for t in sub_trials]), axis=0)
            best_medoid_list = get_medoids_with_distance(sub_trials, s_mean, 1)

            if best_medoid_list:
                path_match = best_medoid_list[0]['path']
                for t in sub_trials:
                    if t['path'] == path_match:
                        medoid_pool.append(t)
                        break

        test_vec = None
        if subject in index:
            for t in index[subject]:
                if t['path'] == test_path:
                    test_vec = t['vec']
                    break

        if test_vec is not None and medoid_pool:
            sel_sz = get_medoids_with_distance(medoid_pool, test_vec, k)
            for x in sel_sz: x['type'] = 'NnMedoidSZ'
    else:
        all_sz_pool = []
        subj_centroids = []
        for sub in other_subjects:
            sub_trials = [t for t in index[sub] if t['label'] == 1 and "class_1" in t['path']]
            if sub_trials:
                all_sz_pool.extend(sub_trials)
                subj_centroids.append(np.mean(np.stack([t['vec'] for t in sub_trials]), axis=0))

        if subj_centroids and all_sz_pool:
            meta_center = np.mean(np.stack(subj_centroids), axis=0)
            sel_sz = get_medoids_with_distance(all_sz_pool, meta_center, k)
            for x in sel_sz: x['type'] = 'RepSZ'

    return sel_ns, sel_sz


# ==========================================
# 2. UTILS & HELPERS
# ==========================================
def get_subject_id(file_path):
    match = re.search(r'(chb\d+|pat\d+|subj\d+)', file_path, re.IGNORECASE)
    return match.group(1) if match else Path(file_path).parent.parent.name


def get_trial_number(file_path):
    match = re.search(r'trial_(\d+)', file_path)
    return int(match.group(1)) if match else -1


def should_process_file(file_path, true_label, dataset_name):
    trial_num = get_trial_number(file_path)
    if trial_num == -1: return False
    if dataset_name == 'NICU':
        return (trial_num % 10 == 0)
    else:
        return (true_label == 1) or (trial_num % 10 == 0)


def convert_path(vision_path, target_type):
    if target_type == "Vision": return vision_path
    return vision_path.replace("/Vision/", f"/{target_type}/").replace("_Vision.png", f"_{target_type}.png")


def load_image_blocking(path):
    if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
    img = Image.open(path)
    img.load()
    return img


def safe_get_text(response):
    """
    Extracts text from response safely.
    Returns: (text, error_message)
    """
    try:
        if not response.candidates:
            return None, "No Candidates (Blocked?)"
        cand = response.candidates[0]
        if cand.finish_reason in [3, 4]:
            return None, f"Safety/Recitation Block (Ratings: {cand.safety_ratings})"
        if not cand.content.parts:
            return None, f"Empty Content (Reason: {cand.finish_reason})"
        return response.text, None
    except Exception as e:
        return None, str(e)


def get_output_paths(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'./results/{args.dataset}'
    os.makedirs(base_dir, exist_ok=True)
    prefix = "RERUN_" if args.rerun_timestamp else ""
    nn_tag = "_NnMedoid" if args.nearest_neighbor else ""
    csv_name = f"{prefix}{args.dataset}_Flash_Type{args.prompt_type}{nn_tag}_N{args.N_fewshots}_{args.selection_mode}_{timestamp}.csv"
    metrics_name = f"{prefix}{args.dataset}_Flash_Metrics_Type{args.prompt_type}{nn_tag}_N{args.N_fewshots}_{args.selection_mode}_{timestamp}.txt"
    return os.path.join(base_dir, csv_name), os.path.join(base_dir, metrics_name)


# ==========================================
# 3. LLM WORKER
# ==========================================
def configure_model(api_key):
    genai.configure(api_key=api_key)
    safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in
              ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
               "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    return genai.GenerativeModel("gemini-3-flash-preview",
                                 safety_settings=safety,
                                 generation_config=genai.GenerationConfig(max_output_tokens=2048, temperature=0.0))


def construct_multimodal_prompt_Z(text_template, test_img_path, examples_with_meta, image_type):
    """
    STRICT CLASS MAPPING for Type Z Prompt:
    - Seizures -> <Nearest Neighbor Example X>
    - Non-Seizures -> <Non-Seizure Example EEG Image X>
    """
    content = []
    replacements = {}
    tag_to_path = {}

    working_text = str(text_template)

    # 0. Load Test Image
    test_img = load_image_blocking(test_img_path)
    replacements["<Test EEG Image>"] = test_img
    tag_to_path["<Test EEG Image>"] = test_img_path

    # --- 1. SEPARATE BY CLASS ---
    sz_examples = [x for x in examples_with_meta if x[0]['label'] == 1]
    ns_examples = [x for x in examples_with_meta if x[0]['label'] == 0]

    # --- 2. MAP SEIZURES TO "NEAREST NEIGHBOR" TAGS ---
    for i, ex_tuple in enumerate(sz_examples):
        prompt_idx = i + 1
        if prompt_idx > 2: break

        ex_dict = ex_tuple[0]
        p = convert_path(ex_dict['path'], image_type)

        tag = f"<Nearest Neighbor Example {prompt_idx}>"

        if tag in working_text:
            try:
                replacements[tag] = load_image_blocking(p)
                tag_to_path[tag] = p
            except:
                pass

    # --- 3. MAP NON-SEIZURES TO "NON-SEIZURE" TAGS ---
    for i, ex_tuple in enumerate(ns_examples):
        prompt_idx = i + 1
        if prompt_idx > 2: break

        ex_dict = ex_tuple[0]
        p = convert_path(ex_dict['path'], image_type)

        tag = f"<Non-Seizure Example EEG Image {prompt_idx}>"

        if tag in working_text:
            try:
                replacements[tag] = load_image_blocking(p)
                tag_to_path[tag] = p
            except:
                pass

    # --- 4. BUILD CONTENT ---
    log_text = working_text

    curr = working_text
    while True:
        earliest_idx = -1
        earliest_tag = None
        for tag in replacements:
            idx = curr.find(tag)
            if idx != -1:
                if earliest_idx == -1 or idx < earliest_idx:
                    earliest_idx = idx
                    earliest_tag = tag

        if earliest_tag is None:
            if curr: content.append(curr)
            break

        content.append(curr[:earliest_idx])
        content.append(replacements[earliest_tag])
        curr = curr[earliest_idx + len(earliest_tag):]

    # Clean log text
    for tag, path_str in tag_to_path.items():
        log_text = log_text.replace(tag, f"[IMG: {path_str}]")

    return content, log_text


async def worker(model, file_path, label, sem, queue, index, args, prompt_txt):
    subject = get_subject_id(file_path)
    fname = Path(file_path).name

    async with sem:
        try:
            # --- 1. SELECTION & PROMPT CONSTRUCTION ---
            ns_ex, sz_ex = await asyncio.to_thread(
                select_examples, index, subject, file_path, args.selection_mode, args.N_fewshots, args.nearest_neighbor
            )

            examples_with_meta = []
            for x in ns_ex: examples_with_meta.append((x, x.get('type', 'NS')))
            for x in sz_ex: examples_with_meta.append((x, x.get('type', 'SZ')))

            all_ex = ns_ex + sz_ex
            used_paths = [x['path'] for x in all_ex]
            dist_str = " | ".join([f"{x.get('type', 'Unknown')}:{x.get('dist', 0.0):.4f}" for x in all_ex])

            content, prompt_log_text = await asyncio.to_thread(
                construct_multimodal_prompt_Z, prompt_txt, file_path, examples_with_meta, args.image_type
            )

            # --- 2. API REQUEST & RESPONSE HANDLING ---
            retries = 0
            final_resp_text = ""
            final_err_msg = ""

            while retries < 3:
                try:
                    # [PHASE A]: Network Transmission
                    if DEBUG_MODE: print(f"[{fname}] Attempt {retries + 1}: Sending Request...")

                    resp = await asyncio.wait_for(model.generate_content_async(content), timeout=180)

                    # [PHASE B]: Response Validation (Disentangled from Network)
                    # If we reach here, the Network Request was SUCCESSFUL.
                    text, err = safe_get_text(resp)

                    if text:
                        # SUCCESS: Valid Content
                        final_resp_text = text
                        final_err_msg = ""  # Clear errors
                        if DEBUG_MODE: print(f"[{fname}] Success.")
                        break  # Exit Retry Loop
                    else:
                        # FAILURE: Request OK, but Content Blocked/Empty
                        # DO NOT RETRY for Safety/Recitation blocks (waste of quota)
                        final_err_msg = f"CONTENT_ERROR: {err}"
                        if "Safety" in str(err) or "Recitation" in str(err):
                            print(f"[{fname}] {final_err_msg}. Stopping retries.")
                            break  # Exit Retry Loop immediately

                        # Only retry if it's a weird empty response that might be transient
                        await asyncio.sleep(2)

                except Exception as e:
                    # [PHASE C]: Network/Server Error
                    err_str = str(e)
                    final_err_msg = f"NETWORK_ERROR: {err_str}"
                    print(f"[{fname}] {final_err_msg}")

                    # Smart Backoff
                    if "429" in err_str:  # Rate Limit
                        await asyncio.sleep(15)
                    elif "500" in err_str or "503" in err_str:  # Server Error
                        await asyncio.sleep(5)
                    else:
                        # Non-transient error (e.g. Invalid Argument), stop retrying
                        break

                retries += 1

            if not final_resp_text:
                await queue.put({'sub': subject, 'file': file_path, 'true': label, 'pred': -1,
                                 'resp': '', 'prompt': prompt_log_text,
                                 'paths': str(used_paths), 'dists': dist_str, 'err': final_err_msg})
                return

            # --- 3. PARSING ---
            pred = -1
            clean = final_resp_text.upper()
            if "FINAL DECISION: SEIZURE" in clean:
                pred = 1
            elif "FINAL DECISION: NON-SEIZURE" in clean:
                pred = 0

            await queue.put({'sub': subject, 'file': file_path, 'true': label, 'pred': pred,
                             'resp': final_resp_text, 'prompt': prompt_log_text,
                             'paths': str(used_paths), 'dists': dist_str, 'err': ''})

        except Exception as e:
            # Catch-all for logic errors (e.g., file not found during selection)
            await queue.put({'sub': subject, 'file': file_path, 'true': label, 'pred': -1,
                             'resp': '', 'prompt': '', 'paths': '[]', 'dists': '', 'err': f"LOGIC_ERROR: {str(e)}"})


# ==========================================
# 4. Calculate Metrics
# ==========================================
def calculate_metrics_report(csv_path, dataset_name):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df = df[df['pred_label'] != -1]
    except:
        return "Error reading CSV."
    if len(df) == 0: return "No results."

    subj_metrics = []
    for s, g in df.groupby('subject_id'):
        yt, yp = g['true_label'], g['pred_label']

        # Weighting logic, considering the downsample
        w = [10 if (dataset_name == 'NICU' or y == 0) else 1 for y in yt]

        # Calculate Weighted BCA
        bca_w = balanced_accuracy_score(yt, yp, sample_weight=w)

        subj_metrics.append({'Subject': s, 'W-BCA': bca_w})

    df_m = pd.DataFrame(subj_metrics).set_index('Subject')
    macro = df_m.mean(numeric_only=True) * 100

    lines = [f"\nDataset: {dataset_name}"]
    lines.append("-" * 35)
    lines.append(f"{'Subject':<15} | {'W-BCA':<6}")
    lines.append("-" * 35)
    for s, r in df_m.iterrows():
        lines.append(f"{s:<15} | {r['W-BCA'] * 100:6.2f}")
    lines.append("-" * 35)
    lines.append(f"AVG (Macro)     | {macro['W-BCA']:6.2f}")
    return "\n".join(lines)


async def writer(q, path):
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not exists:
            w.writerow(['subject_id', 'file_path', 'true_label', 'pred_label', 'raw_response',
                        'prompt_text', 'few_shot_paths', 'few_shot_distances', 'error'])
        while True:
            r = await q.get()
            if r is None: break
            w.writerow([r['sub'], r['file'], r['true'], r['pred'], r['resp'].replace('\n', ' '),
                        r['prompt'], r['paths'], r['dists'], r['err']])
            f.flush()
            q.task_done()


# ==========================================
# 5. MAIN
# ==========================================
async def main(args):
    print(f"--- Prompt: {args.prompt_type} | NN Medoid Mode: {args.nearest_neighbor} ---")

    # --- TEST FILE MODE ---
    if args.test_file:
        p = os.path.join(f'./results/{args.dataset}', args.test_file)
        if os.path.exists(p):
            print(f"Generating report for: {p}")
            print(calculate_metrics_report(p, args.dataset))
        else:
            print(f"Error: File not found at {p}")
        return

    index = load_raw_index(args.dataset)
    model = configure_model(args.api_key)

    try:
        fname = f"prompt-{args.prompt_type}.txt"
        with open(fname) as f:
            prompt = f.read()
    except:
        return print(f"Prompt file {fname} missing.")

    root = os.path.join(args.data_dir, f"{args.dataset}/time-chan/", args.image_type)
    all_files_to_process = []
    print("Scanning...")
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".png"):
                p = os.path.join(r, f)
                l = 1 if "class_1" in p else 0
                if should_process_file(p, l, args.dataset): all_files_to_process.append((p, l))

    csv_path, met_path = get_output_paths(args)

    # --- RERUN LOGIC ---
    done_map = {}
    if args.rerun_timestamp:
        print(f"RERUN MODE: Looking for logs with timestamp {args.rerun_timestamp}...")
        olds = glob.glob(os.path.join(f'./results/{args.dataset}', f"*{args.rerun_timestamp}*.csv"))
        if not olds:
            print("WARNING: No previous files found for this timestamp. Starting fresh.")
        else:
            for o in olds:
                try:
                    t = pd.read_csv(o, on_bad_lines='skip')
                    for _, row in t.iterrows():
                        if int(row['pred_label']) != -1:
                            done_map[row['file_path']] = row
                except Exception as e:
                    print(f"Error reading old CSV {o}: {e}")
            print(f"Recovered {len(done_map)} completed records.")

            # Pre-fill new CSV with old data
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(['subject_id', 'file_path', 'true_label', 'pred_label', 'raw_response',
                            'prompt_text', 'few_shot_paths', 'few_shot_distances', 'error'])
                for _, r in done_map.items():
                    w.writerow([r['subject_id'], r['file_path'], r['true_label'], r['pred_label'],
                                r['raw_response'], r.get('prompt_text', ''), r.get('few_shot_paths', ''),
                                r.get('few_shot_distances', ''), r.get('error', '')])

    # Filter tasks: Only process what isn't in done_map
    tasks = [t for t in all_files_to_process if t[0] not in done_map]

    print(f"Total files: {len(all_files_to_process)}")
    print(f"Already done: {len(done_map)}")
    print(f"Remaining tasks: {len(tasks)}")

    if not tasks:
        print("All tasks completed. Generating report...")
        print(calculate_metrics_report(csv_path, args.dataset))
        return

    sem = asyncio.Semaphore(1)
    q = asyncio.Queue()
    w_task = asyncio.create_task(writer(q, csv_path))

    await tqdm_asyncio.gather(*[worker(model, p, l, sem, q, index, args, prompt) for p, l in tasks])
    await q.put(None)
    await w_task

    rep = calculate_metrics_report(csv_path, args.dataset)
    print(rep)
    with open(met_path, 'w') as f:
        f.write(rep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)  # dataset name
    parser.add_argument('--api_key')  # API key from Gemini API, not required when in test mode when --test_file is given
    parser.add_argument('--data_dir', default="/")  # path to dataset, not required when in test mode when --test_file is given
    parser.add_argument('--image_type', default="VLM")  # Query VLM with "VLM"-type plot which includes the axis, ticks, and channel names. The "Vision"-type plot does not have these, only having the EEG waveforms. See plot_timechan.py
    parser.add_argument('--prompt_type', default="Z")  # default is Z, which is the prompt used/described in paper.
    parser.add_argument('--N_fewshots', type=int, default=2)  # NOTE: if changing this parameter, prompt-Z.txt also have to be changed manually to contain more examples. default 2 PER CLASS = 4 TOTAL.
    parser.add_argument('--selection_mode', default='representative') # Enables Representativeness Metric
    parser.add_argument('--nearest_neighbor', type=str2bool, default=True)  # Enables Similarity Metric
    parser.add_argument('--rerun_timestamp', type=str)  # When given as a path to the .csv, continue querying from breakpoints
    parser.add_argument('--test_file', type=str) # When given as a path to the .csv, only conduct test mode to calculate performance metrics without querying API

    args = parser.parse_args()
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))