# Google map generator

library(plotKML)
library(ggmap)
library(maps)

# Configuration
TRACKS_PATH = "/home/alberto/Dropbox/BSC/GMM/data/real/mallorca/mallorca.csv"
DELIMITER = ";"
HEADER = TRUE
LOCATION = "Mallorca"
RESOLUTION = c(1920, 1080)

# Get map
map <- get_map(location=LOCATION, zoom=9, maptype="satellite", source="google")
map <- ggmap(map)

# Read tracks
tracks <- read.table(TRACKS_PATH, sep=DELIMITER, header=HEADER)

# Add tracks to the map
for (i in 1:dim(tracks)[1]) {
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
  track <- data.frame(lat, lon)
  map <- map + geom_point(data=track, aes(x=lon, y=lat), size=0.5, color="red", alpha=0.5)
}

# Save as png
png(filename="map.png", width=RESOLUTION[1], height=RESOLUTION[2])
plot(map)
dev.off()
