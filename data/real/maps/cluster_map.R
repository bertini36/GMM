# Cluster maps generator

library(plotKML)

# Configuration
TRACKS_PATH ="/home/alberto/Documentos/data/real/porto/results100k/porto_subset_100k_int50_plot.csv"
ASSIGNMENTS_PATH = "/home/alberto/Documentos/data/real/porto/results100k/assignments_plot.csv"
DELIMITER = ";"
HEADER = TRUE
COLORS = c("#0066CC", "#CC0000", "#009933", "#996633", "#9900CC", "#00ffff", "#ff9900", "#336600")
RESOLUTION = c(1920, 1080)

# Read tracks and assignments
tracks <- read.table(TRACKS_PATH, sep=DELIMITER, header=HEADER)
assignments <- read.table(ASSIGNMENTS_PATH, sep=DELIMITER, header=HEADER)

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
ids <- unique(index)

# Global map 
png(filename="clusters.png", width=RESOLUTION[1], height=RESOLUTION[2])
plot(tracks$lon, tracks$lat, type="n", axes=FALSE, xlab="", ylab="", main="", asp=1)
for (i in 1:length(ids)) {
  track <- subset(tracks, index==ids[i])
  lines(track$lon, track$lat, col=paste(COLORS[assignments[ids[i],] + 1], "30", sep=""))
}
dev.off()

# Cluster individuals maps
K <- dim(unique(assignments))[1]
for (i in 0:K-1) {
  png(filename=paste("K", toString(i), ".png", sep=""), width=RESOLUTION[1], height=RESOLUTION[2])
  plot(tracks$lon, tracks$lat, type="n", axes=FALSE, xlab="", ylab="", main="", asp=1)
  for (j in 1:length(ids)) {
    track <- subset(tracks, index==ids[j])
    if (assignments[ids[j],] == i) {
      lines(track$lon, track$lat, col=COLORS[i + 1])
    }
  }
  dev.off()
}