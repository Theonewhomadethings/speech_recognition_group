% Specify the folder containing your audio samples
folder_path = 'E:\speech_coursework2\EEEM030cw2_DevelopmentSet\';
folder = dir(fullfile(folder_path,'*.mp3'));
N_Files = 390;
audio_files=zeros(N_Files);
Fs=zeros(N_Files);
all_coeff=[];
for i=1:length(folder)
    
    
    % Construct the full path to the current audio file
    current_file = fullfile(folder_path, folder(i).name);

    % Read the audio file
    [audio, fs] = audioread(current_file);

    % Extract MFCC features
    coeff=mfcc(audio, fs);

    % Append the MFCC features to the array
    all_coeff = [all_coeff; coeff];
end
%n=length(audio_files);
