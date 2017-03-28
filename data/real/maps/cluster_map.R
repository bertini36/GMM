library(plotKML)

# Configuration
TRACKS_PATH = '/home/alberto/Dropbox/BSC/GMM/data/real/mallorca/mallorca_int.csv'
ASSIGNMENTS_PATH = '/home/alberto/Dropbox/BSC/GMM/inference/pyInference/mixtureOfGaussians/generated/assignments.csv'
DELIMITER = ';'
HEADER = TRUE
COLORS = c("#0066CC30", "#CC000030", "#00993330", "#99663330", "#9900CC30", "#00ffff30", "#ff990030", "#33660030")

# Read tracks
tracks <- read.table(TRACKS_PATH, sep=DELIMITER, header=HEADER)
assignments <- read.table(ASSIGNMENTS_PATH, sep=DELIMITER, header=HEADER)

# Dataframe creation
index <- c()
lat <- c()
lon <- c()
for (i in 1:dim(tracks)[1]) {
  track <- as.character(tracks$Points[i])
  points <- strsplit(track, '], \\[')[[1]]
  points[1] <- strsplit(points[1], '\\[\\[')[[1]][2]
  points[length(points)] <- strsplit(points[length(points)], ']]')[[1]][1]
  index <- c(index, rep(i, length(points)))
  for (j in 1:length(points)) {
    point <- strsplit(points[j], ',')[[1]]
    lat <- c(lat, as.numeric(point[1]))
    lon <- c(lon, as.numeric(point[2]))
  }
}
tracks <- data.frame(cbind(index, lat, lon))

# Map creation
ids <- unique(index)
plot(tracks$lon, tracks$lat, type="n", axes=FALSE, xlab="", ylab="", main="", asp=1)
for (i in 1:length(ids)) {
  track <- subset(tracks, index==ids[i])
  if (assignments[ids[i],] == 0) {
    # lines(track$lon, track$lat, col="#0066CC30")
  } else if (assignments[ids[i],] == 1) {
    # lines(track$lon, track$lat, col="#CC000030")
  } else if (assignments[ids[i],] == 2) {
    # lines(track$lon, track$lat, col="#00993330")
  } else if (assignments[ids[i],] == 3) {
    # lines(track$lon, track$lat, col="#99663330")
  }
  lines(track$lon, track$lat, col=COLORS[assignments[ids[i],] + 1])
}