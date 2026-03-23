import numpy as np
import pandas as pd

def process_all_months(
    dataframe,
    feature_cols,
    meas_cols,
    number_of_months,
    dynamic_threshold,
    stable_keep_step
):
    total_rows = len(dataframe)
    rows_per_month = total_rows // number_of_months

    dataframe = dataframe.copy()
    ref = dataframe["P_I_ref"].values
    
    # 1. GENERATE FEATURES & EVENT IDs
    time_since_change = np.zeros(len(ref), dtype=np.float32)
    jump_direction = np.zeros(len(ref), dtype=np.float32)
    event_ids = np.zeros(len(ref), dtype=np.int32) # New: track event IDs
    
    current_direction = 0.0
    current_event_id = 0

    for i in range(1, len(ref)):
        diff = ref[i] - ref[i - 1]
        if diff == 0:
            time_since_change[i] = time_since_change[i - 1] + 1
            jump_direction[i] = current_direction
        else:
            time_since_change[i] = 0
            current_direction = np.sign(diff)
            jump_direction[i] = current_direction
            current_event_id += 1 # Increment event ID on ref change
        
        event_ids[i] = current_event_id

    dataframe["TimeSinceChange"] = np.log1p(time_since_change)
    dataframe["JumpDirection"] = jump_direction
    dataframe["EventID"] = event_ids # Attach event IDs to dataframe

    # Feature column indexing
    meas_start_idx = len(feature_cols) - len(meas_cols)
    final_feature_cols = (
        feature_cols[:meas_start_idx]
        + ["JumpDirection","TimeSinceChange"]
        + feature_cols[meas_start_idx:]
    )

    all_features_train, all_targets_train = [], []
    all_features_valid, all_targets_valid = [], []
    all_features_test, all_targets_test = [], []
    all_downsampled_data = []

    # 2. MONTHLY PROCESSING
    for m in range(number_of_months):
        start = m * rows_per_month
        end = total_rows if m == number_of_months - 1 else (m + 1) * rows_per_month
        month_df = dataframe.iloc[start:end].reset_index(drop=True)

        measured = month_df[meas_cols].values
        ref_vals = month_df["P_I_ref"].values

        # DOWNSAMPLING LOGIC (Remains the same as your original)
        kept_indices = [0]
        steady_counter = 0
        LOOK_AHEAD = 5
        for i in range(1, len(month_df)):
            ref_changed = abs(ref_vals[i] - ref_vals[i - 1]) > 1e-2
            meas_changed = np.any(np.abs(measured[i] - measured[i - 1]) > dynamic_threshold)
            upcoming_change = False
            if i + LOOK_AHEAD < len(month_df):
                if abs(ref_vals[i + LOOK_AHEAD] - ref_vals[i]) > 1e-2:
                    upcoming_change = True

            if ref_changed or meas_changed or upcoming_change:
                kept_indices.append(i)
                steady_counter = 0
            else:
                steady_counter += 1
                if steady_counter >= stable_keep_step:
                    kept_indices.append(i)
                    steady_counter = 0

        month_down = month_df.iloc[kept_indices].reset_index(drop=True)
        all_downsampled_data.append(month_down)

        # 3. EVENT-BASED SPLITTING (Per Month)
        unique_events = month_down["EventID"].unique()
        #n_events = len(unique_events)
        
        # Calculate event-based split points
        train_event_ids = unique_events[:11]
        valid_event_ids = unique_events[11:13]
        test_event_ids  = unique_events[13:]

        # Filter dataframe rows by event clusters
        df_train = month_down[month_down["EventID"].isin(train_event_ids)]
        df_valid = month_down[month_down["EventID"].isin(valid_event_ids)]
        df_test  = month_down[month_down["EventID"].isin(test_event_ids)]

        # Append to lists
        all_features_train.append(df_train[final_feature_cols].values.astype(np.float32))
        all_targets_train.append(df_train[meas_cols].values.astype(np.float32))
        all_features_valid.append(df_valid[final_feature_cols].values.astype(np.float32))
        all_targets_valid.append(df_valid[meas_cols].values.astype(np.float32))
        all_features_test.append(df_test[final_feature_cols].values.astype(np.float32))
        all_targets_test.append(df_test[meas_cols].values.astype(np.float32))

    # 4. CONCATENATION
    return (
        np.concatenate(all_features_train),
        np.concatenate(all_targets_train),
        np.concatenate(all_features_valid),
        np.concatenate(all_targets_valid),
        np.concatenate(all_features_test),
        np.concatenate(all_targets_test),
        pd.concat(all_downsampled_data, ignore_index=True),
    )