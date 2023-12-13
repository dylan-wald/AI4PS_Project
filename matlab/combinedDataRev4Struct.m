C_PM = 0.0014;

% data to csv
vcData = vc.Data;
swData = sw.Data;
i1Data = i1.Data;
i2Data = i2.Data;
ioutData = iout.Data;
voutData = vout.Data;

combinedData = [vcData swData i1Data i2Data ioutData voutData];
combinedDataSlim = combinedData(1600000:1620000,:);

writematrix(combinedData,'combinedDataRev4.csv');
writematrix(combinedDataSlim,'combinedDataRev4Slim.csv');

