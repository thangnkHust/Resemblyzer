# Step 1:
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from pydub import AudioSegment
import os

#give the file path to your audio file
# audio_file_path = "../Demo/record1_1.wav"
# wav_fpath = Path(audio_file_path)
path_folder = "audio_test"
filename = "Vũ Văn Viện - GĐ Sở Giao thông.mp3"
wav_fpath = Path(path_folder, filename)

wav = preprocess_wav(wav_fpath)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
print("d-vector.....\n", cont_embeds.shape)

# Step 2: Spectral Clustering by Quan Wang
from spectralcluster import SpectralClusterer

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=100,
    p_percentile=0.90,
    gaussian_blur_sigma=1)

labels = clusterer.predict(cont_embeds)

def create_labelling(labels,wav_splits):
    from resemblyzer import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))
    return labelling

labelling = create_labelling(labels,wav_splits)

print(labels)
print("Clustering....\n", labelling)
# temp = 0
# for data in labelling:
#     (id, start, end) = data
#     start = start*1000
#     end = end*1000
#     if not os.path.exists(path_folder + "/test" + id):
#         os.mkdir(path_folder + "/test" + id)
#     if start != end and temp == 0:
#         print('id ' + id)
#         print('start time: ', start)
#         print('end time: ', end)
#         newAudio = AudioSegment.from_mp3(wav_fpath)
#         newAudio = newAudio[start:end]
#         newAudio.export('audio_test/newSong.mp3', format="mp3")
#     temp += 1
