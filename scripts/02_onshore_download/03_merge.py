#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import h5py


#------------------------
# Config
#------------------------

base_dir = "/home/bxd240002/scratch/Archer/onshore_tremor_download"
years = range(2017, 2026)

merge_dir = os.path.join(base_dir, "merge")
os.makedirs(merge_dir, exist_ok=True)

merged_csv_path = os.path.join(merge_dir, "tremor_channels_master_2017-2025.csv")
merged_h5_path = os.path.join(merge_dir, "tremor_raw_master_2017-2025.hdf5")


#------------------------
# Merge CSV
#------------------------

dataframes = []
for year in years:
    csv_path = os.path.join(base_dir, f"tremor_{year}", f"tremor_channels_master_{year}.csv")
    if os.path.exists(csv_path):
        df_year = pd.read_csv(csv_path)
        df_year["source_year"] = year
        dataframes.append(df_year)
    else:
        print(f"[WARN] CSV file not found for year {year}: {csv_path}")

if not dataframes:
    raise FileNotFoundError("No yearly CSV files found to merge.")

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.sort_values(["event_uid", "network", "station", "source_year"], inplace=True)
combined_df.drop_duplicates(subset=["event_uid", "network", "station"], keep="last", inplace=True)
combined_df.drop(columns=["source_year"], inplace=True, errors="ignore")

if "event_start_iso" in combined_df.columns:
    combined_df.sort_values(["event_start_iso", "event_uid", "network", "station"], inplace=True)
else:
    combined_df.sort_values(["event_uid", "network", "station"], inplace=True)


#------------------------
# Merge HDF5
#------------------------

events_in_csv = set(combined_df["event_uid"].astype(str))

with h5py.File(merged_h5_path, "w") as h5_out:
    events_out_group = h5_out.create_group("events")
    h5_out.attrs["kind"] = "tremor_raw_master"
    h5_out.attrs["schema_version"] = "1.0"
    h5_out.attrs["time_window_seconds"] = 600.0

    copied_events = set()

    for year in sorted(years, reverse=True):
        h5_path = os.path.join(base_dir, f"tremor_{year}", f"tremor_raw_master_{year}.hdf5")
        if not os.path.exists(h5_path):
            print(f"[WARN] HDF5 file not found for year {year}: {h5_path}")
            continue

        print(f"Merging events from {h5_path} ...")
        with h5py.File(h5_path, "r") as h5_in:
            if "events" not in h5_in:
                print(f"[WARN] No 'events' group in {h5_path}, skipping.")
                continue

            events_in = h5_in["events"]
            for event_id in events_in:
                if event_id in copied_events:
                    continue
                if event_id not in events_in_csv:
                    print(f"Skipping event {event_id} from {year} (not in any CSV, possibly incomplete)")
                    continue

                h5_in.copy(events_in[event_id], events_out_group, name=event_id)
                copied_events.add(event_id)


#------------------------
# Final Consistency Check
#------------------------

events_copied = copied_events
events_csv_final = set(combined_df["event_uid"].astype(str))
events_missing = events_csv_final - events_copied

if events_missing:
    print(f"[INFO] Removing {len(events_missing)} events from CSV that lacked HDF5 data.")
    combined_df = combined_df[~combined_df["event_uid"].isin(events_missing)].copy()
    events_csv_final = set(combined_df["event_uid"].astype(str))


#------------------------
# Save Outputs
#------------------------

combined_df.to_csv(merged_csv_path, index=False)
print(f"[DONE] Merged CSV saved to {merged_csv_path}")
print(f"[DONE] Merged HDF5 saved to {merged_h5_path}")