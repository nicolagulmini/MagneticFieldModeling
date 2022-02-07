%% show that scalar model and Biot-Savart are equivalent - test the speed
clear

addpath('model')
load('Board_complete_mm.mat') % rememeber that field will be in [A/mm]


Xmin = -150; Xmax = 150; 
Ymin = -150; Ymax = 150; 
Zmin = 100; Zmax = 400; 

%% evaluation points - grid
numPoints_test = 31;
[XX,YY,ZZ] = ndgrid( linspace(Xmin,Xmax,numPoints_test), linspace(Ymin,Ymax,numPoints_test), linspace(Zmin,Zmax,numPoints_test) );
PP_test = [XX(:),YY(:),ZZ(:)]';
numPoints_test = size(PP_test,2);

fluxes_biot_grd = zeros(numPoints_test,3,8);

for ii = 1:numPoints_test
    PP = PP_test(:,ii);
    [Hx,Hy,Hz] = fBiotModel(BF_A,BF_B,PP);
    fluxes_biot_grd(ii,:,:) = [Hx,Hy,Hz]'; % [A/m]
end

% writematrix(fluxes_biot,'fluxes_biot_grid.csv')
% writematrix(PP_test,'position_grid.csv')

%% evaluation points - random
% numPoints_test = 100000;
rng(0)
PP_test(1,:) = -150+300*rand(1,numPoints_test);
PP_test(2,:) = -150+300*rand(1,numPoints_test);
PP_test(3,:) = 100+300*rand(1,numPoints_test);

fluxes_biot_rnd = zeros(numPoints_test,3,8);

for ii = 1:numPoints_test
    PP = PP_test(:,ii);
    [Hx,Hy,Hz] = fBiotModel(BF_A,BF_B,PP);
    fluxes_biot_rnd(ii,:,:) = [Hx,Hy,Hz]'; % [A/m]
end

% writematrix(fluxes_biot,'fluxes_biot_rnd.csv')
% writematrix(PP_test,'position_rnd.csv')



%% plot
plot3(BF_A(1,:),BF_A(2,:),BF_A(3,:))
hold on
plot3([Xmin Xmax Xmax Xmin Xmin],[Ymin Ymin Ymax Ymax Ymin], Zmin+0*[Ymin Ymin Ymax Ymax Ymin],'b-')
plot3([Xmin Xmax Xmax Xmin Xmin],[Ymin Ymin Ymax Ymax Ymin], Zmax+0*[Ymin Ymin Ymax Ymax Ymin],'b-')
plot3(Xmin+0*[Xmin Xmax Xmax Xmin Xmin],[Ymin Ymin Ymax Ymax Ymin], [Zmin Zmax Zmax Zmin Zmin],'b-')
plot3(Xmax+0*[Xmin Xmax Xmax Xmin Xmin],[Ymin Ymin Ymax Ymax Ymin], [Zmin Zmax Zmax Zmin Zmin],'b-')


axis equal
grid on
xlabel('X (mm)')
ylabel('Y (mm)')
zlabel('Z (mm)')
zlim([0 Inf])
