library("rgl")
library(densitycut) # https://bitbucket.org/jerry00/densitycut_dev/src/master/
library(robustbase)

PATH = '/Users/poyuchen/Desktop/UBC/Engineering-Physics/Fifth-Year/Term 1/CPSC 545/benchmark.nosync/benchmark/benchmark-scPhere/example/'

PlotSphere = function(x, cluster, col, density=FALSE, legend=FALSE) {
  if (missing(col)) {
    col = distinct.col
  }
  if (density == FALSE) {
    col.point = AssignLabelColor(col, cluster, 
                                 uniq.label = sort(unique(cluster)))
  } else {
    colours = colorRampPalette((brewer.pal(7, "YlOrRd")))(10)
    FUN = colorRamp(colours)
    
    cluster = (cluster - min(cluster)) / diff(range(cluster))
    col.point = rgb(FUN(cluster), maxColorValue=256)
  }
  plot3d(x[, 1:3], col = col.point, 
         xlim=c(-1, 1), ylim=c(-1, 1), zlim=c(-1, 1), 
         box=FALSE, axes=FALSE, xlab='', ylab='', zlab='')
  
  arrow3d(c(0, -1.35, 0), c(0, 1.35, 0), 
          col = 'gray', s=0.04, type='extrusion', lit=FALSE)
  
  spheres3d(0, 0, 0, lit=FALSE, color='#dbd7d2', 
            alpha=0.6, radius=0.99)
  spheres3d(0, 0, 0, radius=0.9999, lit=FALSE, color='gray', 
            front='lines', alpha=0.6)
  
  if (density == FALSE) {
    id = !duplicated(cluster)
    col.leg = AssignLabelColor(col, cluster)[id]
    leg = cluster[id]
    names(col.leg) = leg
    
    if (legend == TRUE) {
      legend3d("topright", legend = leg, 
               pch = 16, col = col.leg, cex=1, inset=c(0.02)) 
    }
    
    cluster.srt = sort(unique(cluster))
    k.centers = sapply(cluster.srt, function(zz) {
      cat(zz, '\t')
      id = cluster == zz
      center = colMedians(as.matrix(x[id, , drop=FALSE]))
    })
    
    k.centers = t(k.centers)
    
    cluster.size = table(cluster)[as.character(cluster.srt)]
    id = which(cluster.size > 0)
    
    if (length(id) > 0) {
      k.centers = k.centers[id, , drop=FALSE]
      cluster.srt = cluster.srt[id]
    }
    
    k.centers = k.centers / sqrt(rowSums(k.centers^2)) * 1.15
    text3d(k.centers, texts=cluster.srt, col='black')
  }
}

# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download
cell.type = read.delim(paste(PATH, 'data/cd14_monocyte_erythroid_celltype.tsv', sep=''), 
                       header=FALSE, stringsAsFactors = FALSE)[,1]

x = read.delim(paste(PATH, 'demo-out/cd14_mono_eryth_latent_250epoch.tsv', sep=''), 
               sep=' ', header=FALSE)

PlotSphere(x, cell.type)

# you can save the 3d plots as png file or html file
rgl.snapshot(paste(PATH, 'demo-out/cd14_mono_eryth_latent_250epoch_scphere.png', sep=''),  fmt='png')

browseURL(writeWebGL(filename=paste(PATH, 'cd14_mono_eryth_latent_250epoch_sphere.html', sep=''),
                     width=1000))

# People may still want to show the 3D plots in 2D. 
# We can simply convert the 3D cartesian coordinates to spherical coordinates using the car2sph function.

library(sphereplot)
y = car2sph(x)

col = AssignLabelColor(distinct.col, cell.type)
NeatPlot(y[, 1:2], col=col, cex=0.25, 
         cex.axis=1, xaxt='n', yaxt='n')