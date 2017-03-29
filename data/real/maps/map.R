# Map generator

library(plotKML)

# Configuration
TRACKS_PATH = "/home/alberto/Dropbox/BSC/GMM/data/real/mallorca/mallorca.csv"
DELIMITER = ";"
HEADER = TRUE
COLOR = "#0066CC30"
RESOLUTION = c(1920, 1080)

# Read tracks
tracks <- read.table(TRACKS_PATH, sep=DELIMITER, header=HEADER)

# Dataframe creation
index <- c()
lat <- c()
lon <- c()
for (i in 1:dim(tracks)[1]) {
  track <- as.character(tracks$Points[i])
  points <- strsplit(track, "], \\[")[[1]]
  points[1] <- strsplit(points[1], "\\[\\[")[[1]][2]
  points[length(points)] <- strsplit(points[length(points)], "]]")[[1]][1]
  index <- c(index, rep(i, length(points)))
  for (j in 1:length(points)) {
    point <- strsplit(points[j], ",")[[1]]
    lat <- c(lat, as.numeric(point[1]))
    lon <- c(lon, as.numeric(point[2]))
  }
}
tracks <- data.frame(cbind(index, lat, lon))

# Map creation
png(filename="map.png", width=RESOLUTION[1], height=RESOLUTION[2])
ids <- unique(index)
plot(tracks$lon, tracks$lat, type="n", axes=FALSE, xlab="", ylab="", main="", asp=1)
for (i in 1:length(ids)) {
  track <- subset(tracks, index==ids[i])
  lines(track$lon, track$lat, col=COLOR)
}
dev.off()
