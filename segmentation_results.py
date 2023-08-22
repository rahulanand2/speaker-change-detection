import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import roc_auc_score
from model import SSCDModel
import simpleder
from inference import Inference
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os, csv

def pair_files(wav_files, rttm_files, dir_path):
    pairs = []
    for wav in wav_files:
        wav_stem = os.path.splitext(wav)[0].replace('trimmed_','').replace('.Mix-Headset','')
        for rttm in rttm_files:
            rttm_stem = os.path.splitext(rttm)[0].replace('trimmed_','')
            if wav_stem == rttm_stem:
                pairs.append((os.path.join(dir_path, wav), os.path.join(dir_path, rttm)))
    return pairs

def get_file_pairs(dir_path):
    # make sure dir_path is an absolute path
    dir_path = os.path.abspath(dir_path)

    # get list of all files in directory
    files = os.listdir(dir_path)
    wav_files = [f for f in files if f.endswith('.wav')]
    rttm_files = [f for f in files if f.endswith('.rttm')]

    # get list of pairs of .wav and .rttm files with the same base name
    file_pairs = pair_files(wav_files, rttm_files, dir_path)

    # raise an error if no pairs are found
    if not file_pairs:
        raise ValueError(f"No paired files found in directory {dir_path}")

    return file_pairs

# def read_rttm(rttm_file_path):
#     turns = []
#     with open(rttm_file_path, 'r') as file:
#         for line in file:
#             parts = line.split()
#             start_time = float(parts[3])
#             duration = float(parts[4])
#             end_time = start_time + duration
#             speaker_id = parts[7]
#             turns.append((start_time, end_time, speaker_id))
#     turns.sort()
#     speaker_change_points = [turns[i - 1][1] for i in range(1, len(turns)) if turns[i - 1][1] != turns[i][0]]
#     return speaker_change_points

def read_rttm(rttm_file_path):
    """
    Read an RTTM file and return the speaker change points.
    """
    # Parse RTTM into a list of tuples: (start_time, end_time, speaker_id)
    turns = []
    with open(rttm_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            speaker_id = parts[7]
            turns.append((start_time, end_time, speaker_id))

    # Sort turns by start time
    turns.sort()

    speaker_change_points = []
    active_speakers = set()  # To track all active speakers at a given time

    for i in range(1, len(turns)):
        prev_start, prev_end, prev_speaker = turns[i-1]
        curr_start, curr_end, curr_speaker = turns[i]

        # If current turn overlaps with previous, check if a new speaker is involved
        if prev_end > curr_start and prev_speaker != curr_speaker:
            speaker_change_points.append(curr_start)

            # Update active speakers
            active_speakers.add(prev_speaker)
            active_speakers.add(curr_speaker)

        elif prev_end <= curr_start:  # If no overlap
            # Remove previous speaker from active speakers
            active_speakers.discard(prev_speaker)

            # If a new speaker begins speaking, add change point
            if curr_speaker not in active_speakers:
                speaker_change_points.append(curr_start)

            # Update active speakers
            active_speakers.add(curr_speaker)

    return speaker_change_points, turns

# def change_points_to_segments(change_points):
#     change_points.sort()
#     segments = [("speaker", change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)]
#     return segments

def change_points_to_segments(turns_or_change_points, is_ground_truth=False, threshold=0.2):
    """
    Convert turns or change points into segments.
    """
    segments = []
    # Sort the change points if not ground truth
    if not is_ground_truth:
        turns_or_change_points.sort()
    # If handling ground truth with speaker information
    if is_ground_truth:
        active_speakers = set()

        for i in range(1, len(turns_or_change_points)):
            prev_start, prev_end, prev_speaker = turns_or_change_points[i - 1]
            curr_start, curr_end, curr_speaker = turns_or_change_points[i]

            # Skip segments where start is greater than or equal to end
            if prev_start >= curr_start:
                continue

            if prev_end > curr_start and prev_speaker != curr_speaker:
                segments.append((prev_start, curr_start, prev_speaker))
                active_speakers.add(prev_speaker)
                active_speakers.add(curr_speaker)
            elif prev_end <= curr_start:
                active_speakers.discard(prev_speaker)
                if curr_speaker not in active_speakers:
                    segments.append((prev_start, curr_start, prev_speaker))
                active_speakers.add(curr_speaker)

    # If handling predictions without speaker information
    else:
        prev_start = None
        prev_end = None
        for i in range(1, len(turns_or_change_points)):
            start = turns_or_change_points[i - 1]
            end = turns_or_change_points[i]

            # Check if the current start and previous end are within the threshold
            if prev_end is not None and (start - turns_or_change_points[prev_end]) < threshold:
                # Update the end of the previous segment to merge with current
                segments[-1] = (segments[-1][0], end, "speaker")
                continue

            if prev_start is not None and start < prev_start:
                # Handle case where change points are not in order
                # You can add custom logic here, e.g., logging a warning or skipping this iteration
                print(f"Warning: start {start} is smaller than previous start {prev_start}")
            # Use a dummy speaker ID, since predictions don't include speaker information
            if start < end:
                segments.append((start, end, "speaker"))
                prev_end = i
            # segments.append((start, end, "speaker"))

    return segments


def detect_speaker_change_point(audio_file_path):
    model = SSCDModel.load_from_checkpoint("model_segmentation.ckpt")
    inf = Inference(model, scd=True)
    y, sr = librosa.load(audio_file_path)
    t = np.arange(len(y)) / sr
    prediction, activations = inf(audio_file_path, return_scd_points=True)
    speaker_change_activations = activations[:, 1]
    threshold = 0.5
    data_values = np.array(activations.data).flatten()
    time_values = np.arange(activations.sliding_window.start, len(data_values) * activations.sliding_window.step,
                            activations.sliding_window.step)
    # Make sure both arrays have the same length
    time_values = time_values[:len(data_values)]
    frame_times = np.linspace(0, len(activations)*0.01, len(activations))
    change_points = frame_times[1:][(speaker_change_activations[:-1] < threshold) & (speaker_change_activations[1:] >= threshold)]
    change_points = np.concatenate((change_points, frame_times[1:][(speaker_change_activations[:-1] >= threshold) & (speaker_change_activations[1:] < threshold)]))
    return change_points, time_values, data_values, threshold, t, y

def plot_prediction_points(rttm_file_path, change_points_predictions, t, y):
    ground_truth_changes, turns = read_rttm(rttm_file_path)
    vline_range = (np.min(y) / 2, np.max(y) / 2)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(t, y, color='grey', linewidth=0.5, label="Waveform")
    for change in change_points_predictions:
        ax.vlines(x=change, ymin=vline_range[0], ymax=vline_range[1], color='red', linewidth=1, alpha=0.7, label='Change Points' if change == change_points_predictions[0] else "")
    for change in ground_truth_changes:
        ax.vlines(x=change, ymin=vline_range[0], ymax=vline_range[1], color='green', linewidth=1, alpha=0.7, label='Ground Truth' if change == ground_truth_changes[0] else "")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Speaker Change Detection Points')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    output_dir = "/home/rahul/Documents/Outputs/scd_segmentation_plots"
    file_name = os.path.splitext(os.path.basename(rttm_file_path))[0].split("_")[1]
    plt.savefig(os.path.join(output_dir, f"{file_name}{'.png'}"))
    plt.show()
    return ground_truth_changes, turns


def calulate_accuracy_metrics(change_points_predictions, ground_truth, time_values, data_values, threshold, turns):
    ref_segments = change_points_to_segments(turns, is_ground_truth=True)
    hyp_segments = change_points_to_segments(change_points_predictions, is_ground_truth=False)

    # Add an artificial end point to both lists if they don't align perfectly
    if len(ref_segments) != len(hyp_segments):
        max_time = max(ref_segments[-1][2], hyp_segments[-1][2])
        while len(ref_segments) < len(hyp_segments):
            ref_segments.append(("speaker", max_time, max_time))
        while len(hyp_segments) < len(ref_segments):
            hyp_segments.append(("speaker", max_time, max_time))

    ref = [(label, start, end) for start, end, label in ref_segments]
    hyp = [(label, start, end) for start, end, label in hyp_segments]

    ref = [
        (label, float(start), float(end))
        for label, start, end in ref
        if all(isinstance(value, (int, float)) or value.replace('.', '', 1).isdigit() for value in [start, end])
    ]

    hyp = [
        (label, float(start), float(end))
        for label, start, end in hyp
        if all(isinstance(value, (int, float)) or value.replace('.', '', 1).isdigit() for value in [start, end])
    ]
    # hyp = list(set(hyp))
    # Calculate DER using simpleder
    der = simpleder.DER(ref, hyp)

    # # Create binary labels for ground truth
    # binary_labels = np.zeros_like(time_values)
    # for i in range(len(ground_truth) - 1):
    #     start_time = ground_truth[i]
    #     end_time = ground_truth[i + 1]
    #     binary_labels[(time_values >= start_time) & (time_values < end_time)] = 1

    # Create binary predictions
    # binary_predictions = (data_values > threshold).astype(int)
    tolerance_window = 0.5  # You can set this to a value that suits your task
    ref_changes = [segment[1] for segment in ref]
    hyp_changes = [segment[2] for segment in hyp]
    ref_labels = [int(any(abs(time - change) <= tolerance_window for change in ref_changes)) for time in time_values]
    hyp_labels = [int(any(abs(time - change) <= tolerance_window for change in hyp_changes)) for time in time_values]

    # Calculate metrics
    auroc = roc_auc_score(ref_labels, hyp_labels)
    precision = precision_score(ref_labels, hyp_labels)
    recall = recall_score(ref_labels, hyp_labels)
    f1 = f1_score(ref_labels, hyp_labels)

    # Save all metrics in a dictionary
    metrics = {
        'DER': der,
        'AUROC': auroc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    print("Metrics:", metrics)

    return metrics, ref , hyp

def plot_segments(ref_segments, hyp_segments, t, y, rttm_file_path, threshold=0.5):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(t, y, color='grey', linewidth=0.5, label="Waveform")

    # Plot the ground truth speaker change points
    speaker_change_points = []
    prev_speaker, prev_start, prev_end = ref_segments[0]
    speaker_change_points.append(prev_start)

    for seg in ref_segments[1:]:
        try:
            curr_speaker, curr_start, curr_end = seg  # Unpack tuple
            curr_start = float(curr_start)
            curr_end = float(curr_end)
        except (ValueError, TypeError):
            print(f"Invalid segment value: {seg}. Skipping this segment.")
            continue

        if curr_speaker == prev_speaker and (curr_start - prev_end) < threshold:
            # Update the end of the previous segment to merge with current
            speaker_change_points[-1] = curr_end
        else:
            speaker_change_points.append(curr_start)

        prev_speaker, prev_start, prev_end = curr_speaker, curr_start, curr_end

    # Plot speaker change points
    for seg in speaker_change_points:
        ax.axvline(x=seg, color='g',ymin=0.25, ymax=0.75, linestyle='-', linewidth=0.5, label="Ground Truth Change")

    # Parse the speaker turns and plot the speaker change points detected by the model
    model_change_points = []
    active_speakers = set()  # To track all active speakers at a given time
    prev_start, prev_end, prev_label = hyp_segments[0]

    for i in range(1, len(hyp_segments)):  # change turns to hyp_segments
        curr_label, curr_start, curr_end = hyp_segments[i]

        if prev_end > curr_start and prev_label != curr_label:
            if prev_end is not None and (curr_start - prev_end) < threshold:
                model_change_points[-1] = curr_end
            else:
                model_change_points.append(curr_start)

            active_speakers.add(prev_label)
            active_speakers.add(curr_label)

        elif prev_end <= curr_start:  # If no overlap
            active_speakers.discard(prev_label)

            if curr_label not in active_speakers:
                model_change_points.append(curr_start)

            active_speakers.add(curr_label)

        prev_label, prev_start, prev_end = curr_label, curr_start, curr_end

    # Plot the speaker change points detected by the model
    for seg in model_change_points:
        ax.axvline(x=seg, color='r',ymin=0.25, ymax=0.75, linestyle='--', linewidth=0.5, label="Model Change Point")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    output_dir = "/home/rahul/Documents/Outputs/scd_segmentation_plots"
    file_name = os.path.splitext(os.path.basename(rttm_file_path))[0].split("_")[1]
    plt.savefig(os.path.join(output_dir, f"{file_name}{'.png'}"))
    plt.show()


def generate_results(audio_file_path, rttm_file_path):
    change_points_predictions, time_values, data_values, threshold, t, y = detect_speaker_change_point(audio_file_path)
    ground_truth, turns = read_rttm(rttm_file_path)  # plot_prediction_points(rttm_file_path, change_points_predictions, t, y)
    metrics, ref, hyp = calulate_accuracy_metrics(change_points_predictions, ground_truth, time_values, data_values,
                                                  threshold, turns)
    plot_segments(ref, hyp, t, y, rttm_file_path)
    return metrics

def get_processed_files(file_path='processed_files_segmentation.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []

# Function to append processed file to the list
def add_processed_file(file_path, processed_file_path='processed_files_segmentation.txt'):
    with open(processed_file_path, 'a') as file:
        file.write(file_path + '\n')


if __name__ == "__main__":
    dir_path = '/home/rahul/Documents/test_data_trimmed'
    # print(dir_path)
    file_pairs = get_file_pairs(dir_path)

    fieldnames = ['File_name', 'DER', 'Precision', 'Recall', 'F1', 'AUROC']

    # Read the list of already processed files
    processed_files = get_processed_files()

    # Write header row
    with open('/home/rahul/Documents/results_segmentation_scd_true.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for pair in file_pairs:
        audio_file, rttm_file = pair

        # Skip if already processed
        if audio_file in processed_files:
            print(f'Skipping {audio_file}, already processed.')
            continue

        try:
            metrics = generate_results(audio_file, rttm_file)
            print(metrics)
        except Exception as e:
            print(f'Error generating results for file pair ({audio_file}, {rttm_file}): {e}')
            continue

        if not metrics:
            print(f'No results generated for file pair ({audio_file}, {rttm_file})')
            continue

        base_name = os.path.splitext(os.path.basename(audio_file))[0].replace('trimmed_', '')
        print(f'Metrics for file {base_name} generated')

        # Add the file name to the metrics dictionary
        metrics['File_name'] = base_name
        print(f'Metrics for file {base_name} saved in file')
        add_processed_file(audio_file)
        # Write the results to the csv file
        with open('/home/rahul/Documents/results_segmentation_scd_true.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(metrics)