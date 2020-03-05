clear
clc
close all

loadconstants

load('BCI_901.csv')


data = BCI_901;
clear BCI_901

max_l = size(data,1)-1000;

R_TS = find(data(:,TM)==R_marker);
L_TS = find(data(:,TM)==L_marker);
U_TS = find(data(:,TM)==U_marker);
D_TS = find(data(:,TM)==D_marker);

for i=1:3%size(R_TS)
    
    window=data(R_TS(i)-window_pre:R_TS(i)+window_post,:);
    subplot(2,2,1)
    plot (window(:,1:4));
    subplot(2,2,3)
    plot (window(:,TM));
    subplot(2,2,2)
    plot (abs(fft(window(:,1:4))));
    subplot(2,2,4)
    plot(abs(dct(window(:,1:4))));
    
    title (['Window for ' num2str(i) '-th right']);
    pause(5);
    
end


