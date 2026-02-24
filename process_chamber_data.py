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
    "10": [0, 1, 2, 3, 4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "7": [0, 1, 2, 3, 4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
    "23": [0, 1, 2, 3, 4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100],
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
    "Chamber 17": (-60, 100),
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

HSV_THRESHOLDS = {
    "DOABLE": ([35, 50, 50], [85, 255, 255]),      # Green
    "VERIFY": ([15, 50, 50], [35, 255, 255]),      # Yellow
    "NOT DOABLE": ([0, 50, 50], [15, 255, 255]),   # Red (Low)
    "NOT DOABLE_ALT": ([165, 50, 50], [180, 255, 255]) # Red (High)
}

def format_ramp_rate(val):
    if pd.isna(val):
        return val
    val_str = str(val)
    def replacer(match):
        num = float(match.group(0))
        return "{:.1f}".format(num)
    new_val = re.sub(r'\d+(\.\d+)?', replacer, val_str)
    return new_val

def calculate_gap(size_str):
    """
    Calculates the required gap for each dimension axis.
    Formula: ceil(3 + (Dim - 18) * 2 / 31)
    Maps dimension range [18, 49] to gap range [3, 5], clamped at [3, 5].
    """
    if not size_str or size_str == "Unknown":
        return "Unknown"
    
    try:
        dims = [int(v) for v in size_str.split('x')]
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
        return "Unknown"

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

            # Handle temperature parsing (extract numbers or special labels)
            if "UP TO" in temp_raw.upper():
                mapping[chamber]["rates"][100] = rate
            else:
                match = re.search(r'(-?\d+)', temp_raw)
                if match:
                    temp_val = int(match.group(1))
                    mapping[chamber]["rates"][temp_val] = rate
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
        hum_images = [os.path.join(hum_folder, f) for f in os.listdir(hum_folder) if f.endswith('.png')]
    else:
        hum_images = [f for f in os.listdir('.') if f.endswith('.png') and 'RAMP' not in f and 'debug' not in f]
    
    for img_path in hum_images:
        print(f"Processing Humidity: {img_path}...")
        results = process_humidity_image(img_path, debug=True)
        if results:
            all_hum_results.extend(results)
    
    if not all_hum_results:
        print("No humidity images found.")
        return

    hum_df = pd.DataFrame(all_hum_results)

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
            
    unique_chambers = sorted(list(hum_chambers.union(set(standardized_ramp.keys()))))
    
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
        size = chamber_data.get("size", "Unknown")
        gap = calculate_gap(size)

        for temp in TEMP_LABELS_FULL:
            # Determine ramp rate
            ramp_rate = "N/A"
            if temp in chamber_rates:
                ramp_rate = chamber_rates[temp]
            else:
                lower_temps = [t for t in chamber_rates.keys() if t <= temp]
                if lower_temps:
                    ramp_rate = chamber_rates[max(lower_temps)]
                else:
                    higher_temps = [t for t in chamber_rates.keys() if t >= temp]
                    if higher_temps:
                        ramp_rate = chamber_rates[min(higher_temps)]

            # Check if Temp within Chamber's Range first
            ch_range = CHAMBER_RANGES.get(chamber)
            if ch_range and not (ch_range[0] <= temp <= ch_range[1]):
                continue

            for hum in HUM_LABELS:
                # Within range - determine status
                if temp < 10 or temp > 90:
                    # Specific requirement: Chamber 10 at 4°C must be DOABLE
                    if chamber == "Chamber 10" and temp == 4:
                        status = "DOABLE"
                    else:
                        # For out-of-humidity-range temps, check if any image data exists for this temp
                        status_data = hum_df[(hum_df["Chamber"] == chamber) & (hum_df["Temperature"] == temp)]
                        if not status_data.empty:
                            status = status_data["Status"].mode()[0]
                        else:
                            # Default to DOABLE if within range and no image data (common for Temp-only)
                            status = "DOABLE"
                else:
                    # Normal operating range [10, 90] - rely on image data if possible
                    status_row = hum_df[(hum_df["Chamber"] == chamber) & (hum_df["Temperature"] == temp) & (hum_df["Humidity"] == hum)]
                    
                    if not status_row.empty:
                        status = status_row.iloc[0]["Status"]
                    else:
                        # If no image data but it's a Temp-only chamber, it's DOABLE
                        if "Hum" not in CHAMBER_TYPES.get(chamber, ""):
                            status = "DOABLE"
                        else:
                            status = "NOT DOABLE"

                all_rows.append({
                    "Chamber": chamber,
                    "Type": CHAMBER_TYPES.get(chamber, "Unknown"),
                    "Temperature": temp,
                    "Humidity": hum,
                    "Status": status,
                    "Size": size,
                    "Gap": gap,
                    "Ramp_Rate": format_ramp_rate(ramp_rate)
                })


    final_df = pd.DataFrame(all_rows)
    # Ensure reproducibility by sorting
    final_df = final_df.sort_values(by=["Chamber", "Temperature", "Humidity"])
    
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
    
    st.title("🔎 Multi-Point Chamber capability Checker")
    st.markdown("Search for chambers that satisfy multiple environmental configurations simultaneously.")

    # Load data
    file_path = "chamber_complete_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: Could not find {file_path}. Please run the processing first.")
        return

    df = pd.read_csv(file_path)
    df['Status'] = df['Status'].str.strip().str.upper()
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')

    # Session State Initialization
    if 'configs' not in st.session_state:
        st.session_state.configs = [{
            'type': 'Temp/Hum',
            'temp': 25, 
            'hum': 50, 
            'ramp': '1.0C/Min', 
            'size': 'Any', 
            'filter_temp': True, 
            'filter_hum': True
        }]

    # Sidebar for control
    st.sidebar.header("Configuration Management")
    if st.sidebar.button("Add Requirement Step"):
        st.session_state.configs.append({
            'type': 'Temp/Hum',
            'temp': 25, 
            'hum': 50, 
            'ramp': '1.0C/Min', 
            'size': 'Any',
            'filter_temp': True, 
            'filter_hum': True
        })
    
    if st.sidebar.button("Reset All"):
        st.session_state.configs = [{
            'type': 'Temp/Hum',
            'temp': 25, 
            'hum': 50, 
            'ramp': '1.0C/Min', 
            'size': 'Any', 
            'filter_temp': True, 
            'filter_hum': True
        }]
        # Using experimental_rerun would be better but simple reset works too

    # UI for inputs
    st.subheader("📋 Environmental Requirements")
    
    temps_list = sorted(df['Temperature'].dropna().unique())
    hums_list = sorted(df['Humidity'].dropna().unique())
    ramps_list = sorted(df['Ramp_Rate'].dropna().unique())
    sizes_list = ['Any'] + sorted(df['Size'].dropna().unique().tolist())
    types_list = sorted(df['Type'].dropna().unique().tolist())

    if not temps_list or not hums_list:
        st.error("Error: No valid Temperature or Humidity data found in the dataset. Please check the processing logic.")
        return

    # Collect configuration results
    config_results = []

    for i, config in enumerate(st.session_state.configs):
        with st.expander(f"Requirement Set #{i+1}", expanded=True):
            # FIRST ROW: Type selection
            st.session_state.configs[i]['type'] = st.selectbox(
                f"Chamber Type - {i+1}",
                options=types_list,
                index=types_list.index(config['type']) if config['type'] in types_list else 0,
                key=f"type_{i}"
            )
            
            selected_type = st.session_state.configs[i]['type']
            is_hum_available = "Hum" in selected_type
            
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 1.5, 1])
            
            with c1:
                st.session_state.configs[i]['filter_temp'] = st.checkbox(f"Filter Temp (Set {i+1})", value=config['filter_temp'], key=f"f_t_{i}")
                if st.session_state.configs[i]['filter_temp']:
                    st.session_state.configs[i]['temp'] = st.select_slider(
                        f"Temp (°C) - {i+1}", 
                        options=temps_list, 
                        value=config['temp'],
                        key=f"t_{i}"
                    )
            
            with c2:
                if is_hum_available:
                    # Humidity is only relevant if Temperature is within [10, 90]
                    current_temp = st.session_state.configs[i]['temp']
                    hum_relevant = 10 <= current_temp <= 90
                    
                    if hum_relevant:
                        st.session_state.configs[i]['filter_hum'] = st.checkbox(f"Filter Hum (Set {i+1})", value=config['filter_hum'], key=f"f_h_{i}")
                        if st.session_state.configs[i]['filter_hum']:
                            st.session_state.configs[i]['hum'] = st.slider(
                                f"Hum (%) - {i+1}", 
                                min_value=int(min(hums_list)), 
                                max_value=int(max(hums_list)), 
                                value=config['hum'],
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
                st.session_state.configs[i]['size'] = st.selectbox(
                    f"Size - {i+1}", 
                    options=sizes_list, 
                    index=sizes_list.index(config['size']) if config['size'] in sizes_list else 0,
                    key=f"s_{i}"
                )
                if config['size'] != 'Any':
                    st.caption(f"🛡️ Req. Gap: {calculate_gap(config['size'])}")
            
            with c5:
                st.write("") # Spacer
                st.write("")
                if len(st.session_state.configs) > 1:
                    if st.button(f"🗑️ Remove", key=f"del_{i}"):
                        st.session_state.configs.pop(i)
                        st.rerun()

    # Logic: Calculate intersections
    common_doable_chambers = set(df['Chamber'].unique())
    results_per_step = []

    for i, config in enumerate(st.session_state.configs):
        # Base filter by Type
        step_df = df[df['Type'] == config['type']]
        
        # Filter by Ramp
        step_df = step_df[step_df['Ramp_Rate'] == config['ramp']]
        
        # Apply Size filter
        if config['size'] != 'Any':
            step_df = step_df[step_df['Size'] == config['size']]

        actual_hum = config['hum']
        # Double check hum relevance here too
        is_hum_available = "Hum" in config['type']
        is_hum_relevant = is_hum_available and (10 <= config['temp'] <= 90)
        
        if config['filter_hum'] and is_hum_relevant:
            # Find closest hum
            avail = step_df[step_df['Temperature'] == config['temp']]['Humidity'].unique() if config['filter_temp'] else step_df['Humidity'].unique()
            if len(avail) > 0:
                actual_hum = min(avail, key=lambda x: abs(x - config['hum']))
        
        # Apply filters to find DOABLE chambers for this step
        query = step_df.copy()
        if config['filter_temp']:
            query = query[query['Temperature'] == config['temp']]
        if config['filter_hum'] and is_hum_relevant:
            query = query[query['Humidity'] == actual_hum]
        
        doable_for_step = set(query[query['Status'] == 'DOABLE']['Chamber'].unique())
        common_doable_chambers = common_doable_chambers.intersection(doable_for_step)
        
        results_per_step.append({
            'id': i+1,
            'query_df': query,
            'doable_chambers': doable_for_step,
            'chamber_type': config['type'],
            'target_t': config['temp'] if config['filter_temp'] else None,
            'target_h': actual_hum if (config['filter_hum'] and is_hum_relevant) else None,
            'ramp': config['ramp'],
            'filter_desc': f"{config['ramp']}" + (f", {config['temp']}°C" if config['filter_temp'] else "") + (f", {actual_hum}% RH" if (config['filter_hum'] and is_hum_relevant) else "")
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
                st.metric(label="Available Chamber", value=chamber)
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
            else:
                st.warning("No exact matches for this set.")
                
                # --- RECOMMENDATION LOGIC ---
                # Filter by Ramp_Rate AND the requested chamber type
                ramp_doable = df[(df['Ramp_Rate'] == res['ramp']) & (df['Status'] == 'DOABLE') & (df['Type'] == res['chamber_type'])].copy()
                if not ramp_doable.empty:
                    st.info("💡 Nearest 'DOABLE' options for this ramp rate:")
                    
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
                    st.error(f"No DOABLE data at all for {res['ramp']}")

            # Show Dataframe
            display_df = res['query_df'][['Chamber', 'Size', 'Gap', 'Temperature', 'Humidity', 'Status']].copy()
            def color_status(val):
                if val == 'DOABLE': return 'background-color: #d4edda; color: #155724'
                if val == 'VERIFY': return 'background-color: #fff3cd; color: #856404'
                return 'background-color: #f8d7da; color: #721c24'
            st.dataframe(display_df.style.map(color_status, subset=['Status']), width='stretch')

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