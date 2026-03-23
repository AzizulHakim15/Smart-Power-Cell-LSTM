import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
#print("plot_data.py loaded successfully")

# ======================================
# Plot predictions
# ======================================
def plot_predictions(true, pred, target_units, refs=None):
    titles = [
        "Active power (P) - (TSO-SPC)",
        "Reactive power (Q) - (TSO-SPC)",
        "Active power (P) - (SPC-SPC)",
        "Reactive power (Q) - (SPC-SPC)",
    ]
    # Define font sizes here for easy adjustment
    LABEL_SIZE = 16
    TITLE_SIZE = 16
    TICK_SIZE = 16
    n_targets = true.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets), sharex=True)
    if n_targets == 1: axes = [axes]

    for i, ax in enumerate(axes):
        if refs is not None and i< refs.shape[1]:
            ax.plot(refs[:, i], label = "Reference signal", color ='black', linestyle= '--',alpha=1, linewidth=1.5)
        
        ax.plot(true[:, i], label="True response", color ='tab:blue',alpha=0.7)
        ax.plot(pred[:, i], label="Predicted response", color ='tab:orange', linestyle= '--',alpha=0.9)
        
        ax.set_title(titles[i], fontsize=TITLE_SIZE, fontweight='bold')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}"))
        ax.set_ylabel(target_units[i], fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("Time step", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.show()


# ======================================
# Plot Original vs Downsampled
# ======================================
def plot_original_vs_downsampled(original_df, downsampled_df):
    ref_cols = ['P_I_ref', 'Q_I_ref']
    meas_cols = ['P_I_meas', 'Q_I_meas']
    weather_col = 'eta_PV'

    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes1[0].plot(original_df[ref_cols[0]], label='P_I_ref', alpha=0.7)
    axes1[0].plot(original_df[meas_cols[0]], label='P_I_meas', alpha=0.7)
    axes1[0].set_ylabel("P_I"); axes1[0].legend(); axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[1].plot(original_df[ref_cols[1]], label='Q_I_ref', alpha=0.7)
    axes1[1].plot(original_df[meas_cols[1]], label='Q_I_meas', alpha=0.7)
    axes1[1].set_ylabel("Q_I"); axes1[1].legend(); axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[2].plot(original_df[weather_col], label='eta_PV', color='green')
    axes1[2].set_ylabel("eta_PV"); axes1[2].legend(); axes1[2].grid(True, linestyle='--', alpha=0.5)
    axes1[-1].set_xlabel("Time step"); fig1.suptitle("Original Data"); plt.tight_layout(); plt.show()

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes2[0].plot(downsampled_df[ref_cols[0]], label='P_I_ref', alpha=0.7)
    axes2[0].plot(downsampled_df[meas_cols[0]], label='P_I_meas', alpha=0.7)
    axes2[0].set_ylabel("P_I"); axes2[0].legend(); axes2[0].grid(True, linestyle='--', alpha=0.5)
    axes2[1].plot(downsampled_df[ref_cols[1]], label='Q_I_ref', alpha=0.7)
    axes2[1].plot(downsampled_df[meas_cols[1]], label='Q_I_meas', alpha=0.7)
    axes2[1].set_ylabel("Q_I"); axes2[1].legend(); axes2[1].grid(True, linestyle='--', alpha=0.5)
    axes2[2].plot(downsampled_df[weather_col], label='eta_PV', color='green')
    axes2[2].set_ylabel("eta_PV"); axes2[2].legend(); axes2[2].grid(True, linestyle='--', alpha=0.5)
    axes2[-1].set_xlabel("Time step"); fig2.suptitle("Downsampled Data"); plt.tight_layout(); plt.show()


# ======================================
# Plot Train/Valid/Test splits per month
# ======================================
def plot_monthly_splits(downsampled_data, feature_cols, meas_cols, number_of_months):
    total_rows = len(downsampled_data)
    rows_per_month = total_rows // number_of_months

    for m in range(number_of_months):
        start = m * rows_per_month
        end = (m + 1) * rows_per_month if m < number_of_months - 1 else total_rows
        month_df = downsampled_data.iloc[start:end].reset_index(drop=True)

        ref_vals = month_df["P_I_ref"].values
        event_idx = np.where(np.abs(np.diff(ref_vals)) > 1e-6)[0] + 1

        event_starts = list(event_idx)
        event_ends = event_starts[1:] + [len(month_df)]
        event_windows = list(zip(event_starts, event_ends))

        sample_event_id = np.full(len(month_df), -1)
        for eid, (s, e) in enumerate(event_windows):
            sample_event_id[s:e] = eid

        num_events = len(event_windows)
        train_evt = int(0.70 * num_events)
        valid_evt = int(0.85 * num_events)

        train_ids = set(range(0, train_evt))
        valid_ids = set(range(train_evt, valid_evt))
        test_ids  = set(range(valid_evt, num_events))

        train_mask = [sample_event_id[i] in train_ids for i in range(len(month_df))]
        valid_mask = [sample_event_id[i] in valid_ids for i in range(len(month_df))]
        test_mask  = [sample_event_id[i] in test_ids  for i in range(len(month_df))]

        fig, axes = plt.subplots(len(meas_cols), 1, figsize=(12, 4*len(meas_cols)), sharex=True)
        if len(meas_cols) == 1: axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(month_df[meas_cols[i]], label='Original', alpha=0.5)
            ax.plot(month_df[meas_cols[i]].iloc[train_mask], 'o', label='Train', markersize=2)
            ax.plot(month_df[meas_cols[i]].iloc[valid_mask], 'x', label='Valid', markersize=2)
            ax.plot(month_df[meas_cols[i]].iloc[test_mask], '^', label='Test', markersize=2)
            ax.set_ylabel(meas_cols[i])
            ax.set_title(f"Month {m+1} splits")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

        axes[-1].set_xlabel("Time step")
        plt.tight_layout()
        plt.show()



def plot_training_history(train_losses, val_losses, patience=None, counter=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label='Train Loss')
    ax.plot(epochs_range, val_losses, label='Validation Loss')

    # Indicate early stopping if applicable
    if patience is not None and counter is not None and counter >= patience:
        early_stop_epoch = len(train_losses)
        ax.axvline(
            early_stop_epoch,
            linestyle='--',
            label=f'Early Stop Epoch {early_stop_epoch}'
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training & Validation Loss per Epoch')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
