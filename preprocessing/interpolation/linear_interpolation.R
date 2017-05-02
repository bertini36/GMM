# Linear interpolation

# Configuration
TRACKS_PATH = "/home/alberto/Dropbox/BSC/GMM/data/real/mallorca.csv"
OUTPUT_PATH = "/home/alberto/Dropbox/BSC/GMM/data/real/mallorca_test.csv"
N_POINTS_INTERPOLATION = 30
DELIMITER = ";"
HEADER = TRUE

# Read tracks
tracks <- read.table(TRACKS_PATH, sep=DELIMITER, header=HEADER)

# Initialize output
write.table(data.frame(Points=character()), file=OUTPUT_PATH, sep=";", quote=FALSE, col.names=TRUE)

tracks.format <- list()
for (i in 1:50) {
  print(i)
  track <- as.character(tracks$Points[i])
  points <- strsplit(track, "], \\[")[[1]]
  points[1] <- strsplit(points[1], "\\[\\[")[[1]][2]
  points[length(points)] <- strsplit(points[length(points)], "]]")[[1]][1]
  lat <- c()
  lon <- c()
  for (j in 1:length(points)) {
    point <- strsplit(points[j], ",")[[1]]
    lat <- c(lat, as.numeric(point[1]))
    lon <- c(lon, as.numeric(point[2]))
  }
  points.int <- approx(lat, lon, method="linear", n=N_POINTS_INTERPOLATION)
  if (length(lat) > 1) {
    track.format <- list()
    str <- "["
    for (j in 1:N_POINTS_INTERPOLATION) {
      track.format[[j]] <- c(points.int$x[j], points.int$y[j])
      str <- paste(str, paste("[", paste(points.int$x[j], paste(", ", paste(points.int$y[j], "], ", sep=""), sep=""), sep=""), sep=""), sep="")
    }
    str <- substr(str, 1, nchar(str)-1)
    str <- substr(str, 1, nchar(str)-1)
    str <- paste(str, "]", sep="")
    write.table(str, file=OUTPUT_PATH, sep=";", col.names=FALSE, row.names=FALSE, quote=FALSE, append=TRUE)
    tracks.format[[i]] <- track.format
  }
}


