begin
  ; Assuming 'data' is your input dataset
  n = 256  ; Number of points in the dataset
  data = random_normal(0, 1, n)  ; Generate some random data

  ; Perform the FFT
  fftResult = fft(data)

  ; Check the result
  print(fftResult)
end
