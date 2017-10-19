% Function f_PlotMultiCenterSigs.m
% 
% Column Input Matrix
% 
function [s_FigHdl, s_AxesHdl, s_PlotHdl] = f_PlotMultiCenterSigs( ...
    pm_ColMatIn, pv_Scale, ps_UseGlobalLims, ...
    pv_XArray, pv_YLabels)

    if ~exist('pv_Scale', 'var')
        pv_Scale = [];
    end
    
    if ~exist('ps_UseGlobalLims', 'var')
        ps_UseGlobalLims = [];
    end
    
    if ~exist('pv_XArray', 'var')
        pv_XArray = [];
    end
    
    if ~exist('pv_YLabels', 'var')
        pv_YLabels = [];
    end    

    [pm_ColMatIn, v_Y] = f_MatCommonCenter2MultiCenter(pm_ColMatIn, ...
        pv_Scale, ps_UseGlobalLims);
    
    %s_FigHdl = figure;
    s_FigHdl = figure('visible', 'off');
    s_AxesHdl = subplot('Position', [0.1 0.1 0.8 0.8]);
    if isempty(pv_XArray)
        s_PlotHdl = plot(pm_ColMatIn,'k');
    else
        s_PlotHdl = plot(pv_XArray, pm_ColMatIn,'k');
    end
    s_Min = min(pm_ColMatIn(:));
    s_Max = max(pm_ColMatIn(:));
    s_Dis = abs(s_Max - s_Min) * 0.01;
    ylim([(s_Min - s_Dis) (s_Max + s_Dis)]);
    if isempty(pv_XArray)
        xlim([1 size(pm_ColMatIn, 1)]);
    else
        xlim([pv_XArray(1) pv_XArray(end)]);
    end
    set(s_AxesHdl, 'YTick', flipud(v_Y), 'YTickLabel', v_Y);
    if ~isempty(pv_YLabels)
        set(s_AxesHdl, 'YTickLabel', flipud(pv_YLabels(:)));
    end

return;
