import cv2
import numpy as np
import pandas as pd
import os
import re
import sys

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Constants for Humidity Processing
TEMP_LABELS = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
HUM_LABELS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99, 100]
IMG_HUM_COUNT = 29 

HSV_THRESHOLDS = {
    "DOABLE": ([35, 50, 50], [85, 255, 255]),      # Green
    "VERIFY": ([15, 50, 50], [35, 255, 255]),      # Yellow
    "NOT DOABLE": ([0, 50, 50], [15, 255, 255]),   # Red (Low)
    "NOT DOABLE_ALT": ([165, 50, 50], [180, 255, 255]) # Red (High)
}

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

            if chamber not in mapping:
                mapping[chamber] = {}

            # Handle temperature parsing (extract numbers or special labels)
            if "UP TO" in temp_raw.upper():
                mapping[chamber][100] = rate
            else:
                match = re.search(r'(-?\d+)', temp_raw)
                if match:
                    temp_val = int(match.group(1))
                    mapping[chamber][temp_val] = rate
        print(f"Successfully loaded ramp mapping for {len(mapping)} chambers.")
        return mapping
    except Exception as e:
        print(f"Error loading ramp mapping: {e}")
        return {}

def extract_chamber_name(filename):
    basename = os.path.basename(filename)
    match = re.search(r'chamber\s*(\d+)', basename, re.IGNORECASE)
    if match:
        return f"Chamber {match.group(1)}"
    return f"Chamber {basename}"

def process_humidity_image(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    chamber_name = extract_chamber_name(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    debug_img = img.copy() if debug else None

    # Detect any "status" color
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lo, hi) in HSV_THRESHOLDS.items():
        color_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.bitwise_or(mask, color_mask)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    
    def find_main_block(coords, max_val, gap_thresh=10):
        hist = np.histogram(coords, bins=range(max_val + 1))[0]
        nonzero = np.where(hist > 0)[0]
        if len(nonzero) == 0: return 0, max_val
        gaps = np.where(np.diff(nonzero) > gap_thresh)[0]
        blocks = []
        if len(gaps) == 0:
            blocks.append((nonzero[0], nonzero[-1]))
        else:
            start = nonzero[0]
            for g in gaps:
                blocks.append((start, nonzero[g]))
                start = nonzero[g+1]
            blocks.append((start, nonzero[-1]))
        return max(blocks, key=lambda b: b[1] - b[0])

    left, right = find_main_block(xs, img.shape[1])
    top, bottom = find_main_block(ys, img.shape[0])

    cell_w = (right - left) / len(TEMP_LABELS)
    cell_h = (bottom - top) / IMG_HUM_COUNT

    results = []
    for i in range(len(TEMP_LABELS)):
        for j in range(IMG_HUM_COUNT):
            if j == 0: continue 
            cx = int(left + (i + 0.5) * cell_w)
            cy = int(bottom - (j + 0.5) * cell_h)
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
                "Temperature": TEMP_LABELS[i],
                "Humidity": HUM_LABELS[j-1],
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
        results = process_humidity_image(img_path)
        if results:
            all_hum_results.extend(results)
    
    if not all_hum_results:
        print("No humidity images found.")
        return

    hum_df = pd.DataFrame(all_hum_results)

    # 2. Add RAMP Data
    print("Merging RAMP data...")
    ramp_mapping = load_ramp_mapping()
    all_rows = []
    
    # Identify unique chambers for reporting
    total_chambers = hum_df["Chamber"].unique()
    chambers_with_ramp = [c for c in total_chambers if c in ramp_mapping]
    print(f"Chambers with ramp data: {len(chambers_with_ramp)} / {len(total_chambers)}")

    for _, row in hum_df.iterrows():
        chamber = row["Chamber"]
        temp = row["Temperature"]
        
        ramp_rate = "N/A"
        if chamber in ramp_mapping:
            chamber_rates = ramp_mapping[chamber]
            if temp in chamber_rates:
                ramp_rate = chamber_rates[temp]
            else:
                # Find nearest lower temp for range mapping
                lower_temps = [t for t in chamber_rates.keys() if t <= temp]
                if lower_temps:
                    ramp_rate = chamber_rates[max(lower_temps)]
                else:
                    # Fallback to nearest higher if no lower exists
                    higher_temps = [t for t in chamber_rates.keys() if t >= temp]
                    if higher_temps:
                        ramp_rate = chamber_rates[min(higher_temps)]
        
        all_rows.append({
            **row.to_dict(),
            "Ramp_Rate": ramp_rate
        })

    final_df = pd.DataFrame(all_rows)
    # Ensure reproducibility by sorting
    final_df = final_df.sort_values(by=["Chamber", "Temperature", "Humidity"])
    
    output_file = "chamber_complete_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(final_df)} rows to {output_file}")

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
        st.session_state.configs = [{'temp': 25, 'hum': 50, 'ramp': '1.0C/Min', 'filter_temp': True, 'filter_hum': True}]

    # Sidebar for control
    st.sidebar.header("Configuration Management")
    if st.sidebar.button("Add Requirement Step"):
        st.session_state.configs.append({
            'temp': 25, 'hum': 50, 'ramp': '1.0C/Min', 
            'filter_temp': True, 'filter_hum': True
        })
    
    if st.sidebar.button("Reset All"):
        st.session_state.configs = [{'temp': 25, 'hum': 50, 'ramp': '1.0C/Min', 'filter_temp': True, 'filter_hum': True}]
        # Using experimental_rerun would be better but simple reset works too

    # UI for inputs
    st.subheader("📋 Environmental Requirements")
    
    temps_list = sorted(df['Temperature'].dropna().unique())
    hums_list = sorted(df['Humidity'].dropna().unique())
    ramps_list = sorted(df['Ramp_Rate'].dropna().unique())

    # Collect configuration results
    config_results = []

    for i, config in enumerate(st.session_state.configs):
        with st.expander(f"Requirement Set #{i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            
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
                st.session_state.configs[i]['filter_hum'] = st.checkbox(f"Filter Hum (Set {i+1})", value=config['filter_hum'], key=f"f_h_{i}")
                if st.session_state.configs[i]['filter_hum']:
                    st.session_state.configs[i]['hum'] = st.slider(
                        f"Hum (%) - {i+1}", 
                        min_value=int(min(hums_list)), 
                        max_value=int(max(hums_list)), 
                        value=config['hum'],
                        key=f"h_{i}"
                    )
            
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

    # Logic: Calculate intersections
    common_doable_chambers = set(df['Chamber'].unique())
    results_per_step = []

    for i, config in enumerate(st.session_state.configs):
        # Base filter by Ramp
        step_df = df[df['Ramp_Rate'] == config['ramp']]
        
        actual_hum = config['hum']
        if config['filter_hum']:
            # Find closest hum
            avail = step_df[step_df['Temperature'] == config['temp']]['Humidity'].unique() if config['filter_temp'] else step_df['Humidity'].unique()
            if len(avail) > 0:
                actual_hum = min(avail, key=lambda x: abs(x - config['hum']))
        
        # Apply filters to find DOABLE chambers for this step
        query = step_df.copy()
        if config['filter_temp']:
            query = query[query['Temperature'] == config['temp']]
        if config['filter_hum']:
            query = query[query['Humidity'] == actual_hum]
        
        doable_for_step = set(query[query['Status'] == 'DOABLE']['Chamber'].unique())
        common_doable_chambers = common_doable_chambers.intersection(doable_for_step)
        
        results_per_step.append({
            'id': i+1,
            'query_df': query,
            'doable_chambers': doable_for_step,
            'target_t': config['temp'] if config['filter_temp'] else None,
            'target_h': actual_hum if config['filter_hum'] else None,
            'ramp': config['ramp'],
            'filter_desc': f"{config['ramp']}" + (f", {config['temp']}°C" if config['filter_temp'] else "") + (f", {actual_hum}% RH" if config['filter_hum'] else "")
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
                ramp_doable = df[(df['Ramp_Rate'] == res['ramp']) & (df['Status'] == 'DOABLE')].copy()
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
            display_df = res['query_df'][['Chamber', 'Temperature', 'Humidity', 'Status']].copy()
            def color_status(val):
                if val == 'DOABLE': return 'background-color: #d4edda; color: #155724'
                if val == 'VERIFY': return 'background-color: #fff3cd; color: #856404'
                return 'background-color: #f8d7da; color: #721c24'
            st.dataframe(display_df.style.map(color_status, subset=['Status']), use_container_width=True)

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