begin
  ; Define the number of days and simulation years
  nDays = 365
  years_do = 10
  nTotalDays = nDays * years_do
  pi = acos(-1.0)

  ; Create a daily time series from 0 to 2*pi*years_do, similar to Python linspace
  xdaily = fspan(0, 2 * pi * years_do, nTotalDays)

  ; Initialize and generate a sinusoidal time series
  sickos = new(nTotalDays, "float")
  sickos = 0.0
  cycles = (/1, 3, 5, 10, 18, 19, 30/)
  do i = 0, dimsizes(cycles)-1
    sickos = sickos + sin(cycles(i) * xdaily)
  end do

  ;************************************************
  ; Create Lanczos BandPass Filter
  ;************************************************
  ihp    = 2                          ; bpf=>band pass filter
  nWgt   = 201
  sigma  = 1.0                        ; Lanczos sigma
  fca    = 1./100.                    ; Frequency cutoff low
  fcb    = 1./20.                     ; Frequency cutoff high
  wgt    = filwgts_lanczos(nWgt, ihp, fca, fcb, sigma)

  ;************************************************
  ; Apply the Lanczos filter to the time series
  ;************************************************
  filtered_sickos = wgt_runave_leftdim(sickos, wgt, 0)

  ;************************************************
  ; Plotting the original and filtered time series
  ;************************************************
  ; Open a PNG workstation
  wks = gsn_open_wks("png", "filtered_time_series")  ; This will create a PNG file

  ; Plot configurations
  res = True
  res@tiMainString = "Original vs. Filtered Time Series"
  res@xyLineColors = (/"blue", "red"/)
  res@xyLineThicknesses = (/3, 3/)

  ; Create the plot
  plot = gsn_csm_xy(wks, xdaily(0:365), (/sickos(0:365), filtered_sickos(0:365)/), res)

  ; Close the workstation to ensure the file is written
  NhlFrame(wks)
end
