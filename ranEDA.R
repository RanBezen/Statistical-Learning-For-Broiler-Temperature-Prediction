library(car)
library(plotly)
require(cowplot)
require(ggpubr)
library(corrplot)

###########------------------------------------EDA-----------------------------------############
# scaling function
data_scale<-function(data){
  maxs <- apply(data, 2, max) 
  mins <- apply(data, 2, min)
  scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
  return(scaled)
}
#------------------------------upload dataset------------------------------
res<-read.csv('dataset.csv', header = TRUE, stringsAsFactors = FALSE)

set.seed(8)
#-------------------------------feature engineering--------------------------------#
res$time_num <- round(sin(res$time_num*pi),digits = 3)

drops <- c("time","date")
res<-res[ , !(names(res) %in% drops)]
head(res)
str(res)
names(res)
dim(res)
names(res)
#---------------------------------print the sd. and the mean of all parameter
for(col in 1:ncol(res)){
  namee<-names(res)
  paste(namee[col],mean(res[,col]), sd(res[,col]))
  cat(paste(namee[col],mean(res[,col]), sd(res[,col])),'\n')
  namee[col]
  mean(res[,1])
  sd(res[,col])
  
}
#------------------------------------scaling dataset for represent boxplots
res_scale<-data_scale(res)
res$id<- seq(1:dim(res)[1])
res_scale$id<- seq(1:dim(res_scale)[1])
head(res_scale)

boxplot(res_scale[,1:6])

#------------------------------------create color vector by the labels
colors<-res[,ncol(res)-1]
for (i in 1:length(colors)){
  colors[i]<-round(res[i,ncol(res)-1]/0.5)*0.5
}


#--------------------------------------
#  Plots
#--------------------------------------

#----------------------------plot all parameter agains its id 

# V1 - age_sec
p1 <- ggplot(res, aes(x = id, y = age_sec, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p1)
# V2 - face_max_temp   
p2 <- ggplot(res, aes(x = id, y = face_max_temp, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p2)
# V2 - env_temp   
p3 <- ggplot(res, aes(x = id, y = env_temp, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p3)
# V2 - wall_temp   
p4 <- ggplot(res, aes(x = id, y = wall_temp, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p4)

# V2 - quantile1.04   
p5 <- ggplot(res, aes(x = id, y = quantile1.04, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p5)

# V2 - quantile6.25   
p9 <- ggplot(res, aes(x = id, y = quantile6.25, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p9)

# V2 - std.300   
p6 <- ggplot(res, aes(x = id, y = std.300, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p6)

# V2 - std.30   
p7 <- ggplot(res, aes(x = id, y = std.30, colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p7)

# V1 - time_num
p8 <- ggplot(res, aes(x = id, y = time_num,colour=as.factor(colors))) + geom_point() +
  scale_colour_brewer(palette = "Blues") 
ggplotly(p8)

dev.off()
theme_set(theme_cowplot(font_size=12)) # reduce default font size
plot_grid(h1, h2,h3,h4,h5,h6,h7,h8,h9, labels = "AUTO")

plot_grid(p1, p2,p3,p4,p5,p6,p7,p8,p9, labels = "AUTO")
dev.off()
h1<-ggdensity(res, x='age_sec',  fill = "lightgray", add = "mean")
h2<-ggdensity(res, x='face_max_temp',  fill = "lightgray", add = "mean")
h3<-ggdensity(res, x='env_temp',  fill = "lightgray", add = "mean")
h4<-ggdensity(res, x='wall_temp',  fill = "lightgray", add = "mean")
h5<-ggdensity(res, x='quantile1.04',  fill = "lightgray", add = "mean")
h6<-ggdensity(res, x='quantile6.25',  fill = "lightgray", add = "mean")
h7<-ggdensity(res, x='std.30',  fill = "lightgray", add = "mean")
h8<-ggdensity(res, x='std.300',  fill = "lightgray", add = "mean")
h9<-ggdensity(res, x='time_num',  fill = "lightgray", add = "mean")

gridExtra::grid.arrange(h1, h2,h3,h4,h5,h6,h7,h8,h9,nrow = 3,ncol=3)


#----------------------------------------- Plot comparing to class
p <- ggplot(res, aes(x = time_num, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = age_sec, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = face_max_temp, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = env_temp, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = wall_temp, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = quantile1.04, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = quantile6.25, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = std.300, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)

p <- ggplot(res, aes(x = std.30, y = y_temp)) + geom_point() +
  scale_colour_brewer(palette = "RdYlGn") 
ggplotly(p)
#--------------------------------------
#  -----------------------------------------------Correlations
#--------------------------------------
jpeg('correl.jpg')
res_cor <- cor(res[,1:ncol(res)-1],use="pairwise.complete.obs")
corrplot(res_cor, order = "hclust", method = "color")
dev.off()

dev.off()
theme_set(theme_cowplot(font_size=12)) # reduce default font size
plot_grid(p1, p2,p3, labels = "AUTO", ncol = 1)


p1 <- ggplot(res, aes(x = quantile6.25, y = std.300, colour=as.factor(colors)))  + geom_point()+
  scale_colour_brewer(palette = "Blues")
ggplotly(p1)

p2 <- ggplot(res, aes(x = face_max_temp, y = quantile1.04, colour=as.factor(colors))) + geom_point()+
  scale_colour_brewer(palette = "Blues") 
ggplotly(p2)

p3 <- ggplot(res, aes(x = env_temp, y = wall_temp, colour=as.factor(colors))) + geom_point()+
  scale_colour_brewer(palette = "Blues") 
ggplotly(p3)


