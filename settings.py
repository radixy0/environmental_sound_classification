model_file = "model/model.h5"

audio_dir = "data/audio/"
aug_dir = "data/audio_augmented/"

x_path = "data/x_train.npy"
y_path = "data/y_train.npy"

data_folder = "data/test/"
out_folder = "data/generated/"

#training settings
num_classes = 10
learning_rate = 1e-2
decay = 1e-6
momentum = 0.9
epochs = 250
batch_size = 32
patience = 10
lr_patience = 2

#spectrogram settings
imwidth = 450
imheight = 300
NFFT = 512
noverlap = 128

#wav settings
chunk_size = 22050
sr=22050
window_size=4 #in seconds
sounds_per_file = 20
file_len_seconds = 120