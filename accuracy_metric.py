import numpy as np
import matplotlib.pylab as plt
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from inference import Inference
from model import SSCDModel
from sklearn.metrics import roc_auc_score
import os

DO_SCD = True

audio_file_path = "trimmed_ES2011a.Mix-Headset.wav"

print("Loading model")
model = SSCDModel.load_from_checkpoint("/home/rahul/Downloads/model.ckpt")
print("Creating inference object")
inf = Inference(model, scd=DO_SCD)

print("Predicting on audio")
prediction, activations = inf(audio_file_path)

# load ground truth from RTTM file
ground_truth = load_rttm("trimmed_ES2011a.rttm")

# prepare ground truth signal
timestamps = np.arange(activations.sliding_window.start,
                       activations.sliding_window.start + activations.sliding_window.step * len(activations.data),
                       activations.sliding_window.step)
ground_truth_signal = np.zeros_like(activations.data)
for speaker in ground_truth.values():
    for segment in speaker.get_timeline():
        start_frame = int((segment.start - activations.sliding_window.start) / activations.sliding_window.step)
        if start_frame < len(ground_truth_signal):
            ground_truth_signal[start_frame] = 1.0

# Calculate ROC AUC
auc_score = roc_auc_score(ground_truth_signal, activations.data)
print(f"ROC AUC Score: {auc_score}")

print("Plotting results")
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 6))

timestamps = timestamps[:len(activations.data)]  # ensure the lengths match by truncating timestamps
ax[0].plot(timestamps + activations.sliding_window.duration / 2, activations.data, label="Activations")
ax[0].set_ylabel("Activation")
ax[0].legend()

# plot ground truth
change_points = np.where(ground_truth_signal == 1)[0]
ax[1].vlines(timestamps[change_points] + activations.sliding_window.duration / 2, ymin=0, ymax=1, color='red', label="Ground Truth")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Ground Truth")
ax[1].legend()

plt.tight_layout()

# save figure with audio file name
audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]  # extract the file name
plt.savefig(f"{audio_file_name}_acc.png", bbox_inches="tight")

print("Printing prediction")
print(prediction)
