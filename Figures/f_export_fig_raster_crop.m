function fig_handle = f_export_fig_raster_crop(fig_handle)
%F_EXPORT_FIG_RASTER_CROP Crops input axis handle
ppm = get(fig_handle,'PaperPosition');
set(fig_handle,'PaperPosition',[0 0 ppm(3:4)]);
set(fig_handle,'PaperSize',ppm(3:4));
end

