% Writes tests/ref/*.mat: the MATLAB originals run on the real data, which
% tests/test_core_vs_matlab.py compares the Python port against. Run it from
% MATLAB with the package folders on disk (Recon_v1, Calibration_v1, Recon_SWE,
% ColorFlow and Data beside FASTER_Fusion_App). Paths are set at the top.

R   = '/Users/yandongliang/Desktop/Recon_Siemens_GUI_package';
OUT = fullfile(R,'FASTER_Fusion_App','tests','ref');   % where the references go
addpath(fullfile(R,'Recon_v1')); addpath(fullfile(R,'Calibration_v1'));
addpath(fullfile(R,'Recon_SWE')); addpath(fullfile(R,'ColorFlow'));
D = fullfile(R,'Data');
sub = @(A,k) A(:,:,k);

%% ---- colour flow: loader, bar, decode ------------------------------------
cfdir = fullfile(D,'3_ColorFlow_Stepped','2026-09-28_ColorFlow');
ps = loadCFStack(fullfile(cfdir,'Phantom_-2+_v2'));
P = ps.panels(1);
excl = false(size(ps.screen,1), size(ps.screen,2)); excl(P.rows,P.cols) = true;
bar = findFlowBar(ps.screen, excl);
fb = ps.flowBox; box = [fb(1)-P.rows(1)+1 fb(2)-P.rows(1)+1 fb(3)-P.cols(1)+1 fb(4)-P.cols(1)+1];
d = cfDecode(P.rgb, bar, 'Box', box);
k = 43:52;
cf = struct('files',{ps.files},'time_s',ps.time_s,'orderBy',ps.orderBy,'rows',P.rows,'cols',P.cols, ...
    'dz',P.dz_mm,'dx',P.dx_mm,'flowBox',ps.flowBox,'flowPanel',ps.flowPanel, ...
    'rgbsum',sum(double(P.rgb(:))),'rgb48',P.rgb(:,:,:,48), ...
    'barRows',bar.rows,'barCols',bar.cols,'barColors',bar.colors,'barZero',bar.zero, ...
    'box',d.box,'resid',d.resid,'flowFrac',d.flowFrac,'static',d.static,'tint',d.tint, ...
    'k',k,'v',d.v(:,:,k),'mask',d.mask(:,:,k),'grey',d.grey(:,:,k), ...
    'nmask',nnz(d.mask),'vsum',sum(d.v(~isnan(d.v))));
save(fullfile(OUT,'cf.mat'),'-struct','cf','-v7');
fprintf('cf done\n');

%% ---- colour flow recon: 2-D frame, SIVV, box volume via the GUI ----------
f = Recon_CF_v2(fullfile(cfdir,'CAL_2026-09-28_CF_-2+.mat'), fullfile(cfdir,'Phantom_-2+_v2'));
U = f.UserData; U.load(); U.setLat(202); U.setClean('keep',[0.5 8]);
st = U.state(); r = U.sivv();
rec = struct('t',st.rec.t,'grey',st.rec.grey,'zz',st.rec.zz,'yy',st.rec.yy,'pos',st.rec.pos,'ang',st.rec.ang, ...
    'Q',r.prof.Q,'z',r.prof.z,'edge',r.prof.edge,'atQ',r.at.Q,'atZ',r.at.z,'depth',st.sivv.depth, ...
    'rejFrac',st.rejFrac,'lat',202,'keep',[0.5 8]);
U.setView('full',false); U.go(); st = U.state();
rec.volT = st.vol.Velocity.t; rec.volVox = st.vol.Velocity.vox; rec.volIso = st.vol.Velocity.iso;
rec.volLat = st.vol.Velocity.lat;
save(fullfile(OUT,'cfrec.mat'),'-struct','rec','-v7'); delete(f);
fprintf('cfrec done\n');

%% ---- SWE: loader, bars, decode -------------------------------------------
swdir = fullfile(D,'2_SWE_Stepped','2026-09-23_SWE');
ps = loadPositionStack(fullfile(swdir,'Phantom'));
excl = false(size(ps.screen,1), size(ps.screen,2));
for j = 1:numel(ps.panels), excl(ps.panels(j).rows, ps.panels(j).cols) = true; end
bars = findColourBars(ps.screen, excl);
P = ps.panels(end); b = bars(end);
tic; d = swiDecode(P.rgb, b.colors); fprintf('swiDecode %.1fs\n', toc);
k = [1 50 90];
sw = struct('nPanels',numel(ps.panels),'rows1',ps.panels(1).rows,'cols1',ps.panels(1).cols, ...
    'rows2',P.rows,'cols2',P.cols,'nBars',numel(bars),'barColors',b.colors,'barBlue',b.hasBlue, ...
    'barRows',b.rows,'barCols',b.cols,'alpha',d.alpha,'resid',d.resid,'box',d.box,'recovered',d.recovered, ...
    'tint',d.tint,'k',k,'t',d.t(:,:,k),'grey',d.grey(:,:,k),'mask',d.mask(:,:,k),'nmask',nnz(d.mask));
save(fullfile(OUT,'swe.mat'),'-struct','sw','-v7');
fprintf('swe done\n');

%% ---- cine: loader, beamscope sweep, sweep blocks, sector -----------------
cdir = fullfile(D,'1_BMode_Cine','2026-09-14_Siemens_10Vpp');
dd = dir(fullfile(cdir,'Calibration')); dd = dd(~[dd.isdir] & ~startsWith({dd.name},'.'));
sc = loadScanStack(fullfile(dd(1).folder,dd(1).name),'Crop',[369 656]);
temp1 = squeeze(sc.B(:,10,:));
sws = beamscopeSweep(temp1);
cw = max(5, round(round(sws.Tmech/2))); if mod(cw,2)==0, cw = cw+1; end
sws2 = beamscopeSweep(temp1,'ClutterWin',cw);
cine = struct('size',size(sc.B),'dz',sc.dz_mm,'dx',sc.dx_mm,'f1',sc.B(:,:,1),'f300',sc.B(:,:,300), ...
    'Bmean',mean(sc.B,3),'temp1',temp1, ...
    'turns',sws.turns,'turnsSeen',sws.turnsSeen,'Tmech',sws.Tmech,'quality',sws.quality,'Tprior',sws.Tprior, ...
    'nspans',numel(sws.spans),'span1',sws.spans{1},'sym',sws.symCurve, ...
    'turns2',sws2.turns,'quality2',sws2.quality,'nspans2',numel(sws2.spans));
dd = dir(fullfile(cdir,'Phantom_v1')); dd = dd(~[dd.isdir] & ~startsWith({dd.name},'.'));
sc2 = loadScanStack(fullfile(dd(1).folder,dd(1).name),'Crop',[369 656]);
zmm = (0:size(sc2.B,1)-1)*sc2.dz_mm; iz = find(zmm >= 23, 1);
Mdet = squeeze(mean(sc2.B,2));
[blocks,turns,info] = findSweepBlocks(Mdet, iz);
cine.blocks = blocks; cine.bturns = turns; cine.bT = info.T; cine.bslope = info.slope; cine.bclass = info.class;
cine.Mdet = Mdet; cine.iz = iz;
% Recon_v1 recon2D for block 1, fraction map, lateral line 120
c = load(fullfile(cdir,'CAL_2026-09-14_10Vpp_-2+.mat'));
FL = blocks(1,1):blocks(1,2);
nCal = numel(c.FrameLine);
colAxis = 1 + (c.colFit(:) - 1) * (numel(FL)-1)/(nCal-1);
ang = interp1(colAxis, c.thetaFit(:), (1:numel(FL))', 'linear'); valid = ~isnan(ang);
src = double(sc2.B(iz:end,:,FL(valid)));
dmm = zmm(iz:end) - 23; lam = 1540/5.2083e6*1000;
[img,zz,yy] = SectorRecon(src,1:nnz(valid),ang(valid),dmm,lam,120);
cine.img = img; cine.zz = zz; cine.yy = yy; cine.ang = ang(valid);
save(fullfile(OUT,'cine.mat'),'-struct','cine','-v7');
fprintf('cine done\n');

%% ---- GE / Verasonics IQ ---------------------------------------------------
iq = loadScanStack(fullfile(D,'1_BMode_Cine','FASTER-AIR_GE9LD_Verasonics_IQ','Phantom_Dark_v3','IQ_2.dat'));
g = struct('size',size(iq.B),'f1',iq.B(:,:,1),'f50',iq.B(:,:,50),'mean',mean(iq.B(:)),'dz',iq.dz_mm,'dx',iq.dx_mm);
save(fullfile(OUT,'iq.mat'),'-struct','g','-v7');
fprintf('iq done\n');

%% ---- tab snapping + angle fit, through the stepped calibration GUI -------
c = load(fullfile(cfdir,'CAL_2026-09-28_CF_-2+.mat'));
f = Calibration_CF(fullfile(cfdir,'Calibration_-2+')); U = f.UserData; U.load();
U.setLat(c.lat_idx); U.search(c.searchWin(1), c.searchWin(2));
rng(1); clicks = [c.cols(:) + 2*randn(numel(c.cols),1), c.rows(:) + 1.5*randn(numel(c.cols),1)];
clicks = clicks(randperm(size(clicks,1)),:);            % out of order, like a person
for i = 1:size(clicks,1), U.addTab(clicks(i,1), clicks(i,2)); end
st = U.state();
U.setTabs(c.zeroRung, c.negRung); st2 = U.state();
tabs = struct('clicks',clicks,'tabs',st.tabs,'bySpacing',st.tabBySpacing,'temp1',st.temp1, ...
    'lat',c.lat_idx,'searchWin',c.searchWin,'zeroRung',c.zeroRung,'negRung',c.negRung, ...
    'colFit',st2.cal.colFit,'thetaFit',st2.cal.thetaFit,'direction',st2.cal.direction,'col0',st2.cal.col0);
save(fullfile(OUT,'tabs.mat'),'-struct','tabs','-v7'); delete(f);
fprintf('tabs done\n');
