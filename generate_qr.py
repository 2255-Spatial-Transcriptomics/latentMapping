import qrcode
link = 'https://github.com/2255-Spatial-Transcriptomics'
img = qrcode.make(link)
img.save('training.jpg')