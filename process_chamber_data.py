import cv2
import numpy as np
import pandas as pd
import os
import re
import sys
import math

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Constants for Humidity Processing
TEMP_LABELS_FULL = list(range(-100, 181))
HUM_LABELS = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
IMG_HUM_COUNT = 29 

CHAMBER_LEGENDS = {
    "DEFAULT": [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    # These chambers have 5C support in the source images; using 7C here
    # shifts that capability to the wrong temperature in the final CSV.
    "10": [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "7": [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "23": [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "8": [0, 1, 2, 5, 6, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "44": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "45": [0, 1, 5, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
}

CHAMBER_TYPES = {
    "Chamber 2": "Temp/Hum",
    "Chamber 4": "Temp",
    "Chamber 6": "Temp/Hum",
    "Chamber 7": "Temp/Hum",
    "Chamber 8": "Temp/Hum",
    "Chamber 9": "Temp (Halt)",
    "Chamber 10": "Temp/Hum",
    "Chamber 12": "Temp/Hum",
    "Chamber 13": "Temp/Hum",
    "Chamber 14": "Temp/Hum",
    "Chamber 15": "Temp",
    "Chamber 17": "Temp/Alt",
    "Chamber 18": "Temp/Hum",
    "Chamber 19": "Temp/Alt",
    "Chamber 21": "Temp",
    "Chamber 22": "Temp/Hum",
    "Chamber 23": "Temp/Hum",
    "Chamber 25": "Salt Fog",
    "Chamber 28": "Salt Fog",
    "Chamber 31": "Temp",
    "Chamber 32": "Temp (Halt)",
    "Chamber 33": "Temp (Halt)",
    "Chamber 34": "Temp",
    "Chamber 37": "Temp",
    "Chamber 38": "Temp/Hum",
    "Chamber 40": "Temp",
    "Chamber 42": "T-Shock",
    "Chamber 43": "Temp/Hum",
    "Chamber 44": "Temp/Hum",
    "Chamber 45": "Temp/Hum",
    "Chamber 46": "Dust",
    "Chamber 50": "Heat Only",
    "Chamber 51": "T-Shock",
    "Chamber 56": "T-Shock",
    "Chamber 57": "T-Shock",
    "Chamber 60": "Temp",
    "QUV": "UV"
}

CHAMBER_RANGES = {
    "Chamber 2": (-25, 93),
    "Chamber 4": (-100, 170),
    "Chamber 6": (-60, 100),
    "Chamber 7": (-25, 100),
    "Chamber 8": (-60, 100),
    "Chamber 9": (-60, 100),
    "Chamber 10": (-30, 100),
    "Chamber 12": (-60, 100),
    "Chamber 13": (-60, 100),
    "Chamber 14": (-60, 100),
    "Chamber 15": (-60, 150),
    "Chamber 17": (-50, 100),
    "Chamber 18": (-60, 100),
    "Chamber 19": (-35, 100),
    "Chamber 21": (-60, 150),
    "Chamber 22": (-25, 100),
    "Chamber 23": (-20, 100),
    "Chamber 25": (35, 35),
    "Chamber 28": (35, 35),
    "Chamber 31": (-100, 170),
    "Chamber 32": (-60, 85),
    "Chamber 33": (-60, 180),
    "Chamber 34": (-20, 150),
    "Chamber 37": (-60, 100),
    "Chamber 38": (-60, 100),
    "Chamber 40": (-60, 175),
    "Chamber 42": (-60, 180),
    "Chamber 43": (-60, 85),
    "Chamber 44": (-60, 100),
    "Chamber 45": (-60, 100),
    "Chamber 46": (-35, 100),
    "Chamber 50": (20, 340),
    "Chamber 51": (-70, 200),
    "Chamber 56": (-70, 175),
    "Chamber 57": (-70, 175),
    "Chamber 60": (-60, 150),
    "QUV": (30, 60)
}

CHAMBER_POWER = {
    "Chamber 18": "30 KW",
    "Chamber 21": "25 KW",
    "Chamber 6": "15 KW",
    "Chamber 43": "15 KW",
    "Chamber 13": "5 KW",
    "Chamber 44": "5 KW",
    "Chamber 12": "3 KW",
    "Chamber 23": "9 KW"
}

CHAMBER_SIZES = {
    "Chamber 2": "24x24x24",
    "Chamber 4": "18x18x22",
    "Chamber 6": "48x48x48",
    "Chamber 7": "19x19x16",
    "Chamber 8": "30x30x30",
    "Chamber 9": "39x39x39",
    "Chamber 10": "19x19x16",
    "Chamber 12": "36x36x36",
    "Chamber 13": "48x48x54",
    "Chamber 14": "30x30x30",
    "Chamber 15": "40x43x40",
    "Chamber 17": "36x36x36",
    "Chamber 18": "96x144x96",
    "Chamber 19": "36x36x36",
    "Chamber 21": "64x60x60",
    "Chamber 22": "38x38x38",
    "Chamber 23": "48x48x48",
    "Chamber 31": "28x24x26",
    "Chamber 33": "54x54x48",
    "Chamber 34": "28x26x34",
    "Chamber 37": "30x30x30",
    "Chamber 38": "24x24x24",
    "Chamber 40": "24x24x21",
    "Chamber 42": "15x15x15",
    "Chamber 43": "84x86x95",
    "Chamber 44": "48x48x49",
    "Chamber 45": "38x38x38",
    "Chamber 46": "39x39x39",
    "Chamber 51": "25x30x18",
    "Chamber 56": "13.5x9x9.5",
    "Chamber 57": "13.5x9x9.5",
    "Chamber 60": "47x47x47"
}

HSV_THRESHOLDS = {
    "DOABLE": ([35, 50, 50], [85, 255, 255]),      # Green
    "VERIFY": ([15, 50, 50], [35, 255, 255]),      # Yellow
    "NOT DOABLE": ([0, 50, 50], [15, 255, 255]),   # Red (Low)
    "NOT DOABLE_ALT": ([165, 50, 50], [180, 255, 255]) # Red (High)
}

def format_ramp_rate(val):
    if pd.isna(val) or val == "N/A" or val == "":
        return ""
    
    val_str = str(val).strip()
    
    # Check for Ft/Min (Altitude Change)
    if "FT/MIN" in val_str.upper():
        nums = re.findall(r'-?\d+\.?\d*', val_str)
        if nums:
            return f"{nums[0]}Ft/Min"
        return ""

    # Extract all numbers from the string
    nums = re.findall(r'\d+\.?\d*', val_str)
    
    if not nums:
        return ""
    
    # Check if the rate contains a range (e.g., "5.0-6.0")
    if '-' in val_str and len(nums) == 2:
        try:
            val1 = float(nums[0])
            val2 = float(nums[1])
            return f"{val1:.1f}-{val2:.1f}C/Min"
        except ValueError:
            pass
            
    # Default: format the first valid float found
    try:
        num = float(nums[0])
        return f"{num:.1f}C/Min"
    except ValueError:
        return ""

STANDARD_RAMP_OPTIONS = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0]

def format_requested_ramp(rate):
    return f"{float(rate):.1f}C/Min"

def extract_ramp_floor(rate_value):
    """
    Converts a ramp string like '2.0C/Min' or '5.0-6.0C/Min' into the minimum
    supported numeric rate so requested ramps can be treated as 'this fast or faster'.
    """
    if pd.isna(rate_value):
        return np.nan

    rate_str = str(rate_value).strip()
    if not rate_str or "FT/MIN" in rate_str.upper():
        return np.nan

    nums = re.findall(r'-?\d+\.?\d*', rate_str)
    if not nums:
        return np.nan

    try:
        return min(float(num) for num in nums)
    except ValueError:
        return np.nan

def calculate_gap(size_str):
    """
    Calculates the required gap for each dimension axis.
    Formula: ceil(3 + (Dim - 18) * 2 / 31)
    Maps dimension range [18, 49] to gap range [3, 5], clamped at [3, 5].
    Minimum gap is strictly 3x3x3.
    """
    if not size_str or size_str == "Unknown" or size_str == "Any":
        return "3x3x3"
    
    try:
        dims = [float(v) for v in size_str.split('x')]
        gaps = []
        for d in dims:
            # Linear scaling from [18, 49] to [3, 5]
            # Val = 3 + (d - 18) * (5 - 3) / (49 - 18)
            gap_val = 3 + (d - 18) * 2 / 31
            # Clamp to [3, 5] and ceil
            res = math.ceil(max(3, min(5, gap_val)))
            gaps.append(str(res))
        return "x".join(gaps)
    except:
        return "3x3x3"

def parse_size_dimensions(size_str):
    """Convert a size string like '48x48x48' into numeric dimensions."""
    if not size_str or str(size_str).strip() in {"Unknown", "Any", "nan"}:
        return None

    try:
        dims = tuple(float(v) for v in str(size_str).split('x'))
        return dims if len(dims) == 3 else None
    except Exception:
        return None

def size_sort_key(size_str):
    """
    Sort sizes from smallest to largest using volume, then dimensions for ties.
    Unknown sizes are pushed to the end.
    """
    dims = parse_size_dimensions(size_str)
    if dims is None:
        return (float("inf"), float("inf"), float("inf"), float("inf"))

    volume = dims[0] * dims[1] * dims[2]
    return (volume, dims[0], dims[1], dims[2])

def size_meets_or_exceeds(candidate_size, minimum_size):
    """Return True when every candidate dimension meets or exceeds the requested size."""
    candidate_dims = parse_size_dimensions(candidate_size)
    minimum_dims = parse_size_dimensions(minimum_size)
    if candidate_dims is None or minimum_dims is None:
        return False

    return all(candidate_dim >= minimum_dim for candidate_dim, minimum_dim in zip(candidate_dims, minimum_dims))

def normalize_requested_temp(temp_value):
    """
    Round temperatures above 10C up to the next multiple of 5 so they align with
    the dataset's Temp/Hum grid (e.g. 26C -> 30C).
    """
    if pd.isna(temp_value):
        return temp_value

    try:
        temp_value = float(temp_value)
    except (TypeError, ValueError):
        return temp_value

    if temp_value > 10 and temp_value % 5 != 0:
        return int(math.ceil(temp_value / 5.0) * 5)
    return int(temp_value) if float(temp_value).is_integer() else temp_value

def get_matching_step_df(df, config):
    """Apply shared requirement-step filtering before the size constraint."""
    if config['type'] == 'Any':
        step_df = df.copy()
    else:
        step_df = df[df['Type'] == config['type']].copy()

    requested_ramp = extract_ramp_floor(config['ramp'])
    if not pd.isna(requested_ramp):
        step_df = step_df[step_df['Ramp_Rate_Floor'] >= requested_ramp]

    actual_temp = normalize_requested_temp(config['temp'])
    actual_hum = config['hum']
    is_hum_available = step_df['Humidity'].notna().any()
    is_hum_relevant = is_hum_available and (10 <= actual_temp <= 90)

    if config['filter_hum'] and is_hum_relevant:
        avail = step_df[step_df['Temperature'] == actual_temp]['Humidity'].dropna().unique() if config['filter_temp'] else step_df['Humidity'].dropna().unique()
        if len(avail) > 0:
            actual_hum = min(avail, key=lambda x: abs(x - config['hum']))

    query = step_df.copy()
    if config['filter_temp']:
        query = query[query['Temperature'] == actual_temp]
    if config['filter_hum'] and is_hum_relevant:
        query = query[query['Humidity'] == actual_hum]

    return step_df, query, actual_temp, actual_hum, is_hum_relevant

def get_allowed_sizes_for_config(df, config, all_sizes):
    """
    Offer all chamber sizes so the user's selection acts as a minimum acceptable size.
    """
    return ['Any'] + all_sizes, None

def get_temperature_guidance(df, chamber_type, requested_temp):
    """
    Return advisory copy for humidity-capable chamber requests that fall outside
    the supported temperature range.
    """
    if requested_temp is None:
        return None

    if chamber_type == 'Any':
        relevant_df = df[df['Humidity'].notna()]
    else:
        relevant_df = df[(df['Type'] == chamber_type) & df['Humidity'].notna()]

    if relevant_df.empty:
        return None

    min_temp = relevant_df['Temperature'].min()
    max_temp = relevant_df['Temperature'].max()

    if requested_temp > max_temp or requested_temp > 100:
        return (
            f"Humidity cannot be controlled over {int(requested_temp)}C as the water "
            "would keep boiling and create too much humidity. Contact Project Engineers"
        )

    if requested_temp < min_temp or requested_temp < -60:
        return "Contact Test Engineers"

    return None

def format_chamber_range(chamber_detail_df, chamber_type):
    min_temp = chamber_detail_df['Temperature'].min()
    max_temp = chamber_detail_df['Temperature'].max()
    lines = [f"Temp: {min_temp:.0f}°C to {max_temp:.0f}°C"]

    if "Hum" in chamber_type:
        hum_series = chamber_detail_df['Humidity'].dropna()
        if not hum_series.empty:
            lines.append(f"Humidity: {hum_series.min():.0f}% to {hum_series.max():.0f}% RH")

    return lines

def load_ramp_mapping(csv_path="ramp_rates.csv"):
    """Loads ramp rates from CSV and returns a nested dictionary."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using empty mapping.")
        return {}

    try:
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            chamber = str(row["Chamber"]).strip()
            temp_raw = str(row["Temperature"]).strip()
            rate = str(row["Ramp_Rate"]).strip()
            # Read size, default to Unknown if missing
            size = str(row.get("Size", "Unknown")).strip()

            if chamber not in mapping:
                mapping[chamber] = {"rates": {}, "size": size}

            # Handle temperature and altitude parsing
            if "UP TO" in temp_raw.upper():
                continue # Ignore max temp; this is not a ramp rate
            
            # Special handling for altitude markers or conditions
            # e.g., "-60C (SL)", "Climb 0-28k", etc.
            mapping[chamber]["rates"][temp_raw] = rate
        print(f"Successfully loaded ramp mapping for {len(mapping)} chambers.")
        return mapping
    except Exception as e:
        print(f"Error loading ramp mapping: {e}")
        return {}

def extract_chamber_id(filename):
    basename = os.path.basename(filename)
    match = re.search(r'chamber\s*(\d+)', basename, re.IGNORECASE)
    return match.group(1) if match else None

def extract_chamber_name(filename):
    ch_id = extract_chamber_id(filename)
    if ch_id:
        return f"Chamber {ch_id}"
    return f"Chamber {os.path.basename(filename)}"

def process_humidity_image(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    chamber_id = extract_chamber_id(image_path)
    temp_labels = CHAMBER_LEGENDS.get(chamber_id, CHAMBER_LEGENDS["DEFAULT"])
    
    chamber_name = extract_chamber_name(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    debug_img = img.copy() if debug else None

    # Detect Grid Bounds using non-white pixels (including gray NOT DOABLE cells)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, nw = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)
    ys_nw, xs_nw = np.where(nw > 0)
    
    def find_main_block_idx(coords, max_val):
        hist, _ = np.histogram(coords, bins=range(max_val + 1))
        nonzero = np.where(hist > 0)[0]
        if len(nonzero) == 0: return 0, max_val
        gaps = np.where(np.diff(nonzero) > 20)[0]
        blocks = []
        if len(gaps) == 0:
            blocks.append((nonzero[0], nonzero[-1]))
        else:
            start = nonzero[0]
            for g in gaps:
                blocks.append((start, nonzero[g]))
                start = nonzero[g+1]
            blocks.append((start, nonzero[-1]))
        return max(blocks, key=lambda b: np.sum(hist[b[0]:b[1]+1]))

    top, bottom = find_main_block_idx(ys_nw, img.shape[0])
    xs_filtered = xs_nw[(ys_nw >= top) & (ys_nw <= bottom)]
    left, right = find_main_block_idx(xs_filtered, img.shape[1])

    # Detect Grid Lines within status bounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    def find_distinct_lines(proj, start_offset, min_gap=10, thresh_ratio=0.3):
        thresh = np.max(proj) * thresh_ratio
        indices = np.where(proj > thresh)[0]
        lines = []
        if len(indices) > 0:
            curr = indices[0]
            for idx in indices[1:]:
                if idx - curr > min_gap:
                    lines.append(curr + start_offset)
                    curr = idx
            lines.append(curr + start_offset)
        return lines

    v_proj = np.sum(edges[top:bottom, left:right+1], axis=0)
    v_lines = find_distinct_lines(v_proj, left)
    
    h_proj = np.sum(edges[top:bottom, left:right+1], axis=1)
    h_lines = find_distinct_lines(h_proj, top)

    # Use detected lines to find cell centers
    # Temperature (X)
    if len(v_lines) >= len(temp_labels):
        # Prefer the lines closest to the expected count
        col_centers = []
        for i in range(len(temp_labels)):
            if i + 1 < len(v_lines):
                col_centers.append((v_lines[i] + v_lines[i+1]) / 2)
            else:
                # Fallback if we ran out of lines
                col_centers.append(v_lines[-1] + (i - len(v_lines) + 1.5) * (v_lines[-1] - v_lines[-2]))
    else:
        # Uniform fallback
        cell_w = (right - left) / len(temp_labels)
        col_centers = [left + (i + 0.5) * cell_w for i in range(len(temp_labels))]

    # Humidity (Y) - inverted (bottom to top)
    if len(h_lines) >= IMG_HUM_COUNT:
        # Reverse and use from bottom up
        h_lines_rev = sorted(h_lines, reverse=True)
        row_centers = []
        for j in range(IMG_HUM_COUNT):
            if j + 1 < len(h_lines_rev):
                row_centers.append((h_lines_rev[j] + h_lines_rev[j+1]) / 2)
            else:
                row_centers.append(h_lines_rev[-1] - (j - len(h_lines_rev) + 1.5) * (h_lines_rev[-2] - h_lines_rev[-1]))
    else:
        # Uniform fallback
        cell_h = (bottom - top) / IMG_HUM_COUNT
        row_centers = [bottom - (j + 0.5) * cell_h for j in range(IMG_HUM_COUNT)]

    results = []
    for i in range(len(temp_labels)):
        for j in range(IMG_HUM_COUNT):
            # j=0 is usually the 0 baseline or similar, but our labels start at index 0 of HUM_LABELS
            cx = int(col_centers[i])
            cy = int(row_centers[j])
            
            roi = hsv[max(0, cy-2):min(hsv.shape[0], cy+3), max(0, cx-2):min(hsv.shape[1], cx+3)]
            
            status = "NOT DOABLE" 
            max_pixels = 0
            for color, (lo, hi) in HSV_THRESHOLDS.items():
                c_mask = cv2.inRange(roi, np.array(lo), np.array(hi))
                pixel_count = cv2.countNonZero(c_mask)
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    status = color.replace("_ALT", "") 

            results.append({
                "Chamber": chamber_name,
                "Temperature": temp_labels[i],
                "Humidity": HUM_LABELS[j],
                "Status": status
            })

            if debug_img is not None:
                color = (0, 255, 0) if status == "DOABLE" else (0, 255, 255) if status == "VERIFY" else (0, 0, 255)
                cv2.circle(debug_img, (cx, cy), 1, color, -1)

    if debug_img is not None:
        cv2.rectangle(debug_img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.imwrite(f"unified_debug_{os.path.basename(image_path)}", debug_img)

    return results

def main():
    # 1. Process Humidity Images
    all_hum_results = []
    hum_folder = 'chamber_data_folder'
    if os.path.exists(hum_folder):
        hum_images = [
            os.path.join(hum_folder, f)
            for f in os.listdir(hum_folder)
            if re.fullmatch(r'chamber\d+\.png', f, re.IGNORECASE)
        ]
    else:
        hum_images = [
            f for f in os.listdir('.')
            if re.fullmatch(r'chamber\d+\.png', f, re.IGNORECASE)
        ]
    
    for img_path in hum_images:
        print(f"Processing Humidity: {img_path}...")
        results = process_humidity_image(img_path, debug=True)
        if results:
            all_hum_results.extend(results)
    
    if not all_hum_results:
        print("No humidity images found.")
        return

    hum_df = pd.DataFrame(all_hum_results)
    hum_status_lookup = {
        (row["Chamber"], row["Temperature"], row["Humidity"]): row["Status"]
        for _, row in hum_df.iterrows()
    }
    hum_temps_by_chamber = {
        chamber: set(group["Temperature"].tolist())
        for chamber, group in hum_df.groupby("Chamber")
    }

    # 2. Add RAMP Data
    print("Merging RAMP data...")
    ramp_mapping = load_ramp_mapping()
    # Create a master set of all combinations
    all_rows = []
    
    # Identify unique chambers from both sources
    hum_chambers = set(hum_df["Chamber"].unique())
    ramp_chambers = set(ramp_mapping.keys())
    # Standardize ramp chambers to "Chamber X" format if needed
    standardized_ramp = {}
    for r_ch in ramp_chambers:
        if "Chamber" not in r_ch:
            standardized_ramp[f"Chamber {r_ch}"] = r_ch
        else:
            standardized_ramp[r_ch] = r_ch
            
    ALLOWED_TYPES = {"Temp", "Temp/Alt", "Temp/Hum", "T-Shock"}
    all_chambers_raw = hum_chambers.union(set(standardized_ramp.keys())).union(set(CHAMBER_TYPES.keys()))
    
    # Only include chambers that map to the ALLOWED_TYPES
    unique_chambers = sorted([c for c in all_chambers_raw if CHAMBER_TYPES.get(c) in ALLOWED_TYPES])
    
    for chamber in unique_chambers:
        match_chamber = standardized_ramp.get(chamber)
        if not match_chamber:
            # Fallback search
            for r_ch in ramp_mapping:
                if r_ch.strip().lower() == chamber.lower() or f"chamber {r_ch.strip().lower()}" == chamber.lower():
                    match_chamber = r_ch
                    break
        
        chamber_data = ramp_mapping.get(match_chamber, {"rates": {}, "size": "Unknown"}) if match_chamber else {"rates": {}, "size": "Unknown"}
        chamber_rates = chamber_data.get("rates", {})
        # Prioritize hardcoded size if available
        size = CHAMBER_SIZES.get(chamber, chamber_data.get("size", "Unknown"))
        gap = calculate_gap(size)

        # Define Altitude Points for Temp/Alt
        alt_points = [0]
        if CHAMBER_TYPES.get(chamber) == "Temp/Alt":
            if chamber == "Chamber 17":
                alt_points = [0, 28000, 70000]
            elif chamber == "Chamber 19":
                alt_points = [-1500, 70000]

        for alt in alt_points:
            # Determine Altitude Change Rate
            alt_change = ""
            if CHAMBER_TYPES.get(chamber) == "Temp/Alt":
                if chamber == "Chamber 17":
                    if alt == 0: alt_change = "7000Ft/Min" # Target for 0-28k
                    elif alt == 28000: alt_change = "3500Ft/Min" # Target for 28k-70k
                    elif alt == 70000: alt_change = "-12000Ft/Min" # Descent
                elif chamber == "Chamber 19":
                    if alt == -1500: alt_change = "4000Ft/Min" # Target for -1.5k-70k
                    elif alt == 70000: alt_change = "-3500Ft/Min" # Descent (back to 0)

            for temp in TEMP_LABELS_FULL:
                # Determine ramp rate
                ramp_rate = "N/A"
                
                # Check for altitude-specific temp rates first
                condition = "SL" if alt == 0 else "70k"
                spec_key = f"{temp}C ({condition})"
                
                if spec_key in chamber_rates:
                    ramp_rate = chamber_rates[spec_key]
                else:
                    # Fallback to nearest neighbor within the same condition
                    cond_rates = {k: v for k, v in chamber_rates.items() if f"({condition})" in k}
                    if cond_rates:
                        temps_only = {}
                        for k, v in cond_rates.items():
                            m = re.search(r'(-?\d+)', k)
                            if m: temps_only[int(m.group(1))] = v
                        
                        if temps_only:
                            lower = [t for t in temps_only.keys() if t <= temp]
                            if lower: ramp_rate = temps_only[max(lower)]
                            else:
                                higher = [t for t in temps_only.keys() if t >= temp]
                                if higher: ramp_rate = temps_only[min(higher)]
                    
                    # Absolute fallback to generic rates if no condition-specific match
                    if ramp_rate == "N/A":
                        if temp in chamber_rates:
                            ramp_rate = chamber_rates[temp]
                        else:
                            # Original robust fallback
                            m_rates = {}
                            for k, v in chamber_rates.items():
                                m = re.search(r'(-?\d+)', k)
                                if m and '(' not in k: m_rates[int(m.group(1))] = v
                            
                            if m_rates:
                                lower = [t for t in m_rates.keys() if t <= temp]
                                if lower: ramp_rate = m_rates[max(lower)]
                                else:
                                    higher = [t for t in m_rates.keys() if t >= temp]
                                    if higher: ramp_rate = m_rates[min(higher)]

                # Check if Temp within Chamber's Range
                ch_range = CHAMBER_RANGES.get(chamber)
                has_image_temp = temp in hum_temps_by_chamber.get(chamber, set())
                if ch_range and not (ch_range[0] <= temp <= ch_range[1]) and not has_image_temp:
                    continue

                # Select humidity labels based on chamber type
                is_hum_chamber = "Hum" in CHAMBER_TYPES.get(chamber, "")
                current_hum_labels = HUM_LABELS if is_hum_chamber else [None]

                for hum in current_hum_labels:
                    # Within range - determine status
                    status = "NOT DOABLE" # Default

                    if is_hum_chamber and has_image_temp:
                        # Preserve the source image exactly for temperatures that appear
                        # in the chamber capability chart, including humidity-specific
                        # DOABLE/VERIFY bands at low and high temperatures.
                        status = hum_status_lookup.get((chamber, temp, hum), "NOT DOABLE")
                    elif temp < 10 or temp > 90:
                        # Outside the charted humidity-temperature grid, treat an in-range
                        # temperature as generically doable when no image-backed point exists.
                        status = "DOABLE"
                    else:
                        # Temp/Hum chambers without source data for this point remain unsupported.
                        ch_type = CHAMBER_TYPES.get(chamber, "")
                        if "Hum" not in ch_type:
                            status = "DOABLE"
                        else:
                            status = "NOT DOABLE"

                    # Clean up altitude for non-Temp/Alt chambers
                    is_alt_chamber = CHAMBER_TYPES.get(chamber) == "Temp/Alt"
                    display_alt = alt if is_alt_chamber else ""
                    display_alt_change = alt_change if is_alt_chamber else ""

                    all_rows.append({
                        "Chamber": chamber,
                        "Type": CHAMBER_TYPES.get(chamber, "Unknown"),
                        "Temperature": temp,
                        "Humidity": hum if is_hum_chamber else "",
                        "Altitude": display_alt,
                        "Altitude_Change": display_alt_change,
                        "Power": CHAMBER_POWER.get(chamber, "2 KW"),
                        "Status": status,
                        "Size": size,
                        "Gap": gap,
                        "Ramp_Rate": format_ramp_rate(ramp_rate)
                    })


    final_df = pd.DataFrame(all_rows)
    # Ensure reproducibility by sorting
    final_df = final_df.sort_values(by=["Chamber", "Altitude", "Temperature", "Humidity"])
    
    # Globally replace any N/A strings with empty strings
    final_df = final_df.replace("N/A", "")
    
    output_file = "chamber_complete_data.csv"
    final_df.to_csv(output_file, index=False)
    
    # Also update chamber_data.csv to keep it in sync
    final_df.to_csv("chamber_data.csv", index=False)
    
    print(f"Successfully saved {len(final_df)} rows to {output_file} and chamber_data.csv")

def run_streamlit_ui():
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit not found. Please install with 'pip install streamlit'.")
        return

    st.set_page_config(page_title="Multi-Point Chamber capability Checker", layout="wide")
    
    st.title("🔎 QL Chamber Checker")
    st.markdown("Search for chambers that satisfy multiple environmental configurations simultaneously.")

    # Load data
    file_path = "chamber_complete_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: Could not find {file_path}. Please run the processing first.")
        return

    df = pd.read_csv(file_path, dtype={"Altitude": "str", "Altitude_Change": "str"})
    df['Status'] = df['Status'].str.strip().str.upper()
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')

    # Session State Initialization
    if 'configs' not in st.session_state:
        st.session_state.configs = [{
            'type': 'Temp/Hum',
            'temp': 25, 
            'hum': 50, 
            'ramp': 'Any',
            'size': 'Any', 
            'filter_temp': True, 
            'filter_hum': True
        }]

    shared_type = st.session_state.configs[0]['type']
    shared_size = st.session_state.configs[0]['size']
    for config in st.session_state.configs:
        config['type'] = shared_type
        config['size'] = shared_size

    # Sidebar for control
    st.sidebar.header("Configuration Management")
    if st.sidebar.button("Add Requirement Step"):
        st.session_state.configs.append({
            'type': st.session_state.configs[0]['type'],
            'temp': 25, 
            'hum': 50, 
            'ramp': 'Any',
            'size': st.session_state.configs[0]['size'],
            'filter_temp': True, 
            'filter_hum': True
        })
    
    if st.sidebar.button("Reset All"):
        st.session_state.configs = [{
            'type': 'Temp/Hum',
            'temp': 25, 
            'hum': 50, 
            'ramp': 'Any',
            'size': 'Any', 
            'filter_temp': True, 
            'filter_hum': True
        }]
        # Using experimental_rerun would be better but simple reset works too

    # UI for inputs
    temps_list = sorted(df['Temperature'].dropna().unique())
    hums_list = sorted(df['Humidity'].dropna().unique())
    ramps_list = ['Any'] + [format_requested_ramp(rate) for rate in STANDARD_RAMP_OPTIONS]
    all_sizes = sorted(df['Size'].dropna().unique().tolist(), key=size_sort_key)
    types_list = ['Any'] + sorted(df['Type'].dropna().unique().tolist())

    df = df.copy()
    df['Ramp_Rate_Floor'] = df['Ramp_Rate'].apply(extract_ramp_floor)

    if not temps_list or not hums_list:
        st.error("Error: No valid Temperature or Humidity data found in the dataset. Please check the processing logic.")
        return

    st.subheader("🏗️ Chamber Setup")
    setup_col1, setup_col2 = st.columns(2)
    type_help = "Shared across all requirement sets. Use Reset All to start over with a different type."

    with setup_col1:
        st.session_state.configs[0]['type'] = st.selectbox(
            "Chamber Type",
            options=types_list,
            index=types_list.index(st.session_state.configs[0]['type']) if st.session_state.configs[0]['type'] in types_list else 0,
            key="shared_type",
            help=type_help
        )

    with setup_col2:
        shared_size_options, _ = get_allowed_sizes_for_config(df, st.session_state.configs[0], all_sizes)
        if st.session_state.configs[0]['size'] not in shared_size_options:
            st.session_state.configs[0]['size'] = 'Any'
        st.session_state.configs[0]['size'] = st.selectbox(
            "Minimum Chamber Size",
            options=shared_size_options,
            index=shared_size_options.index(st.session_state.configs[0]['size']) if st.session_state.configs[0]['size'] in shared_size_options else 0,
            key="shared_size"
        )
        if st.session_state.configs[0]['size'] != 'Any':
            st.caption(f"Showing chambers sized {st.session_state.configs[0]['size']} or larger")
            st.caption(f"🛡️ Req. Gap: {calculate_gap(st.session_state.configs[0]['size'])}")

    shared_type = st.session_state.configs[0]['type']
    shared_size = st.session_state.configs[0]['size']
    for config in st.session_state.configs:
        config['type'] = shared_type
        config['size'] = shared_size

    st.subheader("📋 Environmental Requirements")
    guidance_container = st.container()
    temp_guidance_messages = []

    for i, config in enumerate(st.session_state.configs):
        with st.expander(f"Requirement Set #{i+1}", expanded=True):
            selected_type = st.session_state.configs[i]['type']
            is_hum_available = selected_type == "Any" or "Hum" in selected_type
            st.caption(f"Chamber Type: {selected_type}")
            if st.session_state.configs[i]['size'] != 'Any':
                st.caption(f"Minimum Chamber Size: {st.session_state.configs[i]['size']} or larger")
            
            c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
            
            with c1:
                st.session_state.configs[i]['filter_temp'] = st.checkbox(f"Filter Temp (Set {i+1})", value=config['filter_temp'], key=f"f_t_{i}")
                if st.session_state.configs[i]['filter_temp']:
                    st.session_state.configs[i]['temp'] = st.number_input(
                        f"Temp (°C) - {i+1}", 
                        min_value=int(min(temps_list)), 
                        max_value=int(max(temps_list)), 
                        value=int(config['temp']),
                        step=1,
                        key=f"t_{i}"
                    )
                    rounded_temp = normalize_requested_temp(st.session_state.configs[i]['temp'])
                    if rounded_temp != st.session_state.configs[i]['temp']:
                        st.caption(f"Using {rounded_temp}°C for matching")
                    guidance = get_temperature_guidance(
                        df,
                        selected_type,
                        st.session_state.configs[i]['temp']
                    )
                    if guidance:
                        temp_guidance_messages.append((i + 1, guidance))
            
            with c2:
                if is_hum_available:
                    # Humidity is only relevant if Temperature is within [10, 90]
                    current_temp = st.session_state.configs[i]['temp']
                    hum_relevant = 10 <= current_temp <= 90
                    
                    if hum_relevant:
                        st.session_state.configs[i]['filter_hum'] = st.checkbox(f"Filter Hum (Set {i+1})", value=config['filter_hum'], key=f"f_h_{i}")
                        if st.session_state.configs[i]['filter_hum']:
                            st.session_state.configs[i]['hum'] = st.number_input(
                                f"Hum (%) - {i+1}", 
                                min_value=int(min(hums_list)), 
                                max_value=int(max(hums_list)), 
                                value=int(config['hum']),
                                step=1,
                                key=f"h_{i}"
                            )
                    else:
                        st.session_state.configs[i]['filter_hum'] = False
                        st.info("ℹ️ Humidity N/A for this Temp range")
                else:
                    st.session_state.configs[i]['filter_hum'] = False
                    st.info(f"ℹ️ Humidity N/A for {selected_type}")
            
            with c3:
                st.session_state.configs[i]['ramp'] = st.selectbox(
                    f"Ramp Rate - {i+1}", 
                    options=ramps_list, 
                    index=ramps_list.index(config['ramp']) if config['ramp'] in ramps_list else 0,
                    key=f"r_{i}"
                )

            with c4:
                st.write("") # Spacer
                st.write("")
                if len(st.session_state.configs) > 1:
                    if st.button(f"🗑️ Remove", key=f"del_{i}"):
                        st.session_state.configs.pop(i)
                        st.rerun()

    with guidance_container:
        for step_id, message in temp_guidance_messages:
            st.warning(f"Requirement Set #{step_id}: {message}")

    # Logic: Calculate intersections
    common_doable_chambers = set(df['Chamber'].unique())
    results_per_step = []

    for i, config in enumerate(st.session_state.configs):
        step_df, query, actual_temp, actual_hum, is_hum_relevant = get_matching_step_df(df, config)

        # Apply Size filter
        if config['size'] != 'Any':
            step_df = step_df[step_df['Size'].apply(lambda size: size_meets_or_exceeds(size, config['size']))]
            query = query[query['Size'].apply(lambda size: size_meets_or_exceeds(size, config['size']))]
        
        doable_for_step = set(query[query['Status'] == 'DOABLE']['Chamber'].unique())
        common_doable_chambers = common_doable_chambers.intersection(doable_for_step)

        filter_parts = []
        if config['type'] != 'Any':
            filter_parts.append(config['type'])
        if config['ramp'] != 'Any':
            filter_parts.append(config['ramp'])
        if config['filter_temp']:
            filter_parts.append(f"{actual_temp}°C")
        if config['filter_hum'] and is_hum_relevant:
            filter_parts.append(f"{actual_hum}% RH")
        if config['size'] != 'Any':
            filter_parts.append(f"{config['size']} or larger")
        
        results_per_step.append({
            'id': i+1,
            'query_df': query,
            'doable_chambers': doable_for_step,
            'chamber_type': config['type'],
            'minimum_size': config['size'],
            'target_t': actual_temp if config['filter_temp'] else None,
            'target_h': actual_hum if (config['filter_hum'] and is_hum_relevant) else None,
            'ramp': config['ramp'],
            'filter_desc': ", ".join(filter_parts) if filter_parts else "Any"
        })


    st.divider()

    # SECTION 1: Master Results
    st.header("🏆 Master Results: Common Suitable Chambers")
    if len(st.session_state.configs) > 1:
        st.markdown(f"Chambers that satisfy **ALL {len(st.session_state.configs)}** requirement sets:")
    
    if common_doable_chambers:
        cols = st.columns(min(len(common_doable_chambers), 5))
        for j, chamber in enumerate(sorted(list(common_doable_chambers))):
            with cols[j % 5]:
                chamber_detail_df = df[df['Chamber'] == chamber]
                chamber_type = chamber_detail_df['Type'].iloc[0]
                chamber_size = chamber_detail_df['Size'].iloc[0]
                chamber_power = chamber_detail_df['Power'].iloc[0]
                chamber_port = ""
                if 'Port' in chamber_detail_df.columns:
                    chamber_port = str(chamber_detail_df['Port'].iloc[0]).strip()

                st.metric(label="Available Chamber", value=chamber)
                if chamber_port and chamber_port.lower() != "nan":
                    st.caption(f"Port: {chamber_port}")
                for detail_line in [f"Size: {chamber_size}", *format_chamber_range(chamber_detail_df, chamber_type), f"Power: {chamber_power}"]:
                    st.caption(detail_line)
                st.success("DOABLE for all sets")
    else:
        st.error("No single chamber satisfies all configurations simultaneously.")
        if len(st.session_state.configs) > 1:
            st.info("Check the individual breakdowns below to see which chambers are closest for each step.")

    st.divider()

    # SECTION 2: Individual Breakdowns
    st.header("📊 Individual Set Breakdowns")
    
    for res in results_per_step:
        with st.expander(f"Results for Set #{res['id']} ({res['filter_desc']})", expanded=False):
            if res['doable_chambers']:
                st.write(f"**Suitable Chambers:** {', '.join(sorted(list(res['doable_chambers'])))}")
                # Results Header
                st.subheader(f"✅ Found {len(res['doable_chambers'])} Potential Chambers")
                
                # Filter full data for display
                display_cols = ["Chamber", "Type", "Temperature", "Humidity", "Altitude", "Altitude_Change", "Power", "Status", "Size", "Ramp_Rate"]
                
                # Use query_df to get only matching rows, then collapse to 1 row per chamber
                results_df = res['query_df'][res['query_df']['Chamber'].isin(res['doable_chambers'])].copy()
                
                # Collapsing Logic: aggregate altitude info if it's a Temp/Alt search result
                has_alt = any(results_df["Type"] == "Temp/Alt")
                
                if has_alt:
                    # Grouping columns - everything except the values we want to list
                    group_cols = [c for c in display_cols if c not in ["Altitude", "Altitude_Change"]]
                    
                    # Convert to string for comma-joining
                    results_df['Altitude'] = results_df['Altitude'].astype(str)
                    results_df['Altitude_Change'] = results_df['Altitude_Change'].fillna("N/A").astype(str)
                    
                    # Aggregate unique values for Altitude and Altitude_Change
                    results_df = results_df.groupby(group_cols).agg({
                        'Altitude': lambda x: ", ".join(sorted(set(x), key=lambda val: int(val) if val.replace('-', '').isdigit() else val)),
                        'Altitude_Change': lambda x: ", ".join(sorted(set(x)))
                    }).reset_index()
                else:
                    # Just drop duplicates for simple chambers
                    results_df = results_df.drop_duplicates(subset=["Chamber"])
                    display_cols = [c for c in display_cols if c not in ["Altitude", "Altitude_Change"]]

                st.dataframe(results_df[display_cols].sort_values("Chamber"), width="stretch")
            else:
                st.warning("No exact matches for this set.")
                
                # --- RECOMMENDATION LOGIC ---
                # Filter by the selected chamber specs while respecting Any selections
                requested_ramp = extract_ramp_floor(res['ramp'])
                ramp_doable = df[df['Status'] == 'DOABLE'].copy()
                if res['chamber_type'] != 'Any':
                    ramp_doable = ramp_doable[ramp_doable['Type'] == res['chamber_type']]
                if not pd.isna(requested_ramp):
                    ramp_doable = ramp_doable[ramp_doable['Ramp_Rate_Floor'] >= requested_ramp]
                if res['minimum_size'] != 'Any':
                    ramp_doable = ramp_doable[ramp_doable['Size'].apply(lambda size: size_meets_or_exceeds(size, res['minimum_size']))]
                if not ramp_doable.empty:
                    st.info("💡 Nearest 'DOABLE' options that match the selected specs:")
                    
                    req_t = res['target_t'] if res['target_t'] is not None else 0
                    req_h = res['target_h'] if res['target_h'] is not None else 0
                    
                    def calc_dist(row):
                        d_t = (row['Temperature'] - req_t)**2 if res['target_t'] is not None else 0
                        d_h = (row['Humidity'] - req_h)**2 if res['target_h'] is not None else 0
                        return np.sqrt(d_t + d_h)

                    ramp_doable['distance'] = ramp_doable.apply(calc_dist, axis=1)
                    min_dist = ramp_doable['distance'].min()
                    closest = ramp_doable[ramp_doable['distance'] == min_dist].drop_duplicates(subset=['Temperature', 'Humidity'])
                    
                    for _, opt in closest.iterrows():
                        opt_t = opt['Temperature']
                        opt_h = opt['Humidity']
                        
                        # Find all chambers for this specific closest point
                        chambers_for_opt = ramp_doable[
                            (ramp_doable['Temperature'] == opt_t) & 
                            (ramp_doable['Humidity'] == opt_h)
                        ]['Chamber'].unique()
                        
                        st.markdown(f"- **{opt_t}°C / {opt_h}% RH** (Distance: {min_dist:.1f})")
                        st.write(f"  Available in: {', '.join(sorted(list(chambers_for_opt)))}")
                else:
                    st.error("No DOABLE data found for the selected filters.")

            # Show Dataframe
            display_df = res['query_df'][['Chamber', 'Size', 'Gap', 'Temperature', 'Humidity', 'Status']].copy()
            # Collapse to 1 row per chamber for the summary view
            display_df = display_df.drop_duplicates(subset=["Chamber"])
            
            def color_status(val):
                if val == 'DOABLE': return 'background-color: #d4edda; color: #155724'
                if val == 'VERIFY': return 'background-color: #fff3cd; color: #856404'
                return 'background-color: #f8d7da; color: #721c24'
            st.dataframe(display_df.style.map(color_status, subset=['Status']), width='stretch')

    # SECTION 3: Detailed Chamber Specifications
    st.divider()
    st.header("🔍 Detailed Chamber Specification Viewer")
    st.markdown("Select a specific chamber to see its full ratings, power performance, and environmental limits.")
    
    all_chambers = sorted(df['Chamber'].unique())
    selected_chamber_detail = st.selectbox("Choose a Chamber for Full Specs", options=all_chambers, key="detail_spec_select")
    
    if selected_chamber_detail:
        chamber_detail_df = df[df['Chamber'] == selected_chamber_detail]
        c_type = chamber_detail_df['Type'].iloc[0]
        c_power = chamber_detail_df['Power'].iloc[0]
        c_size = chamber_detail_df['Size'].iloc[0]
        
        # Display key metadata in columns
        m1, m2, m3 = st.columns(3)
        m1.metric("Chamber Type", c_type)
        m2.metric("Power Rating", c_power)
        m3.metric("Size", c_size)
        
        # Specific details for Alt Chambers
        if c_type == "Temp/Alt":
            st.info(f"🚀 **Altitude Performance Details for {selected_chamber_detail}**")
            # Show a summary table of altitude points
            alt_summary = chamber_detail_df[['Altitude', 'Altitude_Change']].drop_duplicates()
            st.table(alt_summary)
            st.write("These rates apply to climb/descent within the respective altitude segments.")
        
        # Temperature and Humidity Limits
        min_temp = chamber_detail_df['Temperature'].min()
        max_temp = chamber_detail_df['Temperature'].max()
        st.write(f"**Temperature Limits:** {min_temp}°C to {max_temp}°C")
        
        if "Hum" in c_type:
            min_hum = chamber_detail_df['Humidity'].min()
            max_hum = chamber_detail_df['Humidity'].max()
            st.write(f"**Humidity Range:** {min_hum}% to {max_hum}% RH")

        with st.expander(f"Full Data Rows for {selected_chamber_detail}", expanded=False):
            st.dataframe(chamber_detail_df, width="stretch")

if __name__ == "__main__":
    # Modern Streamlit detection
    is_streamlit = False
    if STREAMLIT_AVAILABLE:
        try:
            from streamlit.runtime import exists
            if exists():
                is_streamlit = True
        except ImportError:
            if 'streamlit' in sys.modules:
                is_streamlit = True
    
    if is_streamlit:
        run_streamlit_ui()
    else:
        main()
