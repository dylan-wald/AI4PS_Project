% C_PM = 0.0014;

% data to csv
vcData = vc.Data;
swData = sw.Data;
i1Data = i1.Data;
i2Data = i2.Data;
ioutData = iout.Data;
voutData = vout.Data;

vcSqueeze = squeeze(vcData);

combinedCapCheck = [vcData swData i1Data i2Data ioutData voutData];
combinedSlim = combinedCapCheck(1630000:1650000,:);
writematrix(combinedSlim,'completeDataBypassSwitchV2.csv')


%%
xPlot = 0:Ts_Power:(.2-Ts_Power);
figure(1);
plot(xPlot,combinedCapFirst0(:,28));
ylim([-2250 2250]);
title('V_{out}')
xlabel('Time (s)')
ylabel('Voltage (V)')

figure(2);
subplot(2,2,1);
yyaxis left
plot(xPlot,combinedCapFirst0(:,5));
ylabel('Voltage (V)')
yyaxis right
plot(xPlot,combinedCapFirst0(:,17));
title('V_{C1}')
xlabel('Time (s)')
ylabel('Switching Signal')
ylim([-0.2 1.2]);

subplot(2,2,2);
yyaxis left
plot(xPlot,combinedCapFirst0(:,6));
ylabel('Voltage (V)')
yyaxis right
plot(xPlot,combinedCapFirst0(:,19));
title('V_{C2}')
xlabel('Time (s)')
ylabel('Switching Signal')
ylim([-0.2 1.2]);

subplot(2,2,3);
yyaxis left
plot(xPlot,combinedCapFirst0(:,7));
ylabel('Voltage (V)')
yyaxis right
plot(xPlot,combinedCapFirst0(:,21));
title('V_{C3}')
xlabel('Time (s)')
ylabel('Switching Signal')
ylim([-0.2 1.2]);

subplot(2,2,4);
yyaxis left
plot(xPlot,combinedCapFirst0(:,8));
ylabel('Voltage (V)')
yyaxis right
plot(xPlot,combinedCapFirst0(:,23));
title('V_{C4}')
xlabel('Time (s)')
ylabel('Switching Signal')
ylim([-0.2 1.2]);

figure(3);
plot(xPlot,combinedCapFirst0(:,25));
hold on
plot(xPlot,combinedCapFirst0(:,26));
plot(xPlot,combinedCapFirst0(:,27));
title('MMC Currents')
xlabel('Time (s)')
ylabel('Current (A)')
% ylim([-42 42]);
legend('i_{1}','i_2','i_{out}');
hold off