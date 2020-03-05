% speechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)
% computes speech spectrograms for the files in the datastore ads.
% segmentDuration is the total duration of the speech clips (in seconds),
% frameDuration the duration of each spectrogram frame, hopDuration the
% time shift between each spectrogram frame, and numBands the number of
% frequency bands.

function X = speechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing speech spectrograms...");

fs        = 256;
FFTLength = 1024*8;
persistent filterBank
if isempty(filterBank)
   filterBank = designAuditoryFilterBank(fs,'FrequencyScale','mel',...
                                            'FFTLength',FFTLength,...
                                            'NumBands',numBands,...
                                            'FrequencyRange',[1,50]);
end

numHops = ceil((segmentDuration - frameDuration)/hopDuration);
numFiles = length(ads.Files);
x=read(ads);
X = zeros([numBands,numHops,size(x,2),numFiles],'single');
ads.reset();
for i = 1:numFiles
    
    x = read(ads);
    for channel=1:size(x,2)
        frameLength = round(frameDuration*fs);
        hopLength = round(hopDuration*fs);

        [~,~,~,spec] = spectrogram(x(:,channel),hann(frameLength,'periodic'),frameLength - hopLength,FFTLength,'onesided');
        spec = filterBank * spec;

        % If the spectrogram is less wide than numHops, then put spectrogram in
        % the middle of X.
        w = size(spec,2);
        left = floor((numHops-w)/2)+1;
        ind = left:left+w-1;
        X(:,ind,channel,i) = spec;
    end
    if mod(i,100) == 0
        disp("Processed " + i + " files out of " + numFiles)
    end
    
end

disp("...done");

end