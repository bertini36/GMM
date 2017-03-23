library(plotKML)
library(ggmap)
library(maps)

TRACKS_PATH = '/home/alberto/Escritorio/test.csv'

# Porto map
map <- get_map(location='Porto', zoom=12, maptype='satellite', source='google')
map <- ggmap(map)

# Read tracks
tracks <- read.table(TRACKS_PATH, sep=',', header=TRUE)

# Add tracks to the map
for (i in 1:dim(tracks)[1]) {
  track <- as.character(tracks[i,]$POLYLINE)
  points <- strsplit(track, '],\\[')[[1]]
  points[1] <- strsplit(points[1], '\\[\\[')[[1]][2]
  points[length(points)] <- strsplit(points[length(points)], ']]')[[1]][1]
  lat <- c()
  lon <- c()
  for (j in 1:length(points)) {
    point <- strsplit(points[j], ',')[[1]]
    lat <- c(lat, as.numeric(point[1]))
    lon <- c(lon, as.numeric(point[2]))
  }
  track <- data.frame(lat, lon)
  map <- map + geom_point(data=track, aes(x=lat, y=lon), size=0.5, color="red", alpha=0.5)
}

plot(map)

# Save as png
png(filename='map.png')
plot(map)
dev.off()
