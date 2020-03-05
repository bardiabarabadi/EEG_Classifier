% backgroundSpectrograms(ads,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands)
% calculates numBkgClips spectrograms of background clips taken from the
% audio files in the |ads| datastore. Approximately the same number of
% clips is taken from each audio file. Before calculating spectrograms, the
% function rescales each audio clip with a factor sampled from a
% log-uniform distribution in the range given by volumeRange.
% segmentDuration is the total duration of the speech clips (in seconds),
% frameDuration the duration of each spectrogram frame, hopDuration the
% time shift between each spectrogram frame, and numBands the number of
% frequency bands.

function Xbkg = backgroundSpectrograms(ads,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing background spectrograms...");

fs        = 256;
FFTLength = 1024*8;
persistent filterBank
if isempty(filterBank)
    filterBank = designAuditoryFilterBank(fs,'FrequencyScale','bark',...
        'FFTLength',FFTLength,...
        'NumBands',numBands,...
        'FrequencyRange',[1,50]);
end

logVolumeRange = log10(volumeRange);

numBkgFiles = numel(ads.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));

numHops = ceil(segmentDuration/hopDuration) - 2;
Xbkg = zeros(numBands,numHops,1,numBkgClips,'single');

ind = 1;
for count = 1:numBkgFiles
    
    wave = read(ads);
    
    frameLength = frameDuration*fs;
    hopLength = hopDuration*fs;
    
    for j = 1:numClipsPerFile(count)
        indStart =  randi(numel(wave)-fs);
        logVolume = logVolumeRange(1) + diff(logVolumeRange)*rand;
        volume = 10^logVolume;
        x = wave(indStart:indStart+fs-1)*volume;
        x = max(min(x,1),-1);
        
        [~,~,~,spec] = spectrogram(x,hann(frameLength,'periodic'),frameLength - hopLength,FFTLength,'onesided');
        Xbkg(:,:,:,ind) = filterBank * spec;

        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end

disp("...done");

end