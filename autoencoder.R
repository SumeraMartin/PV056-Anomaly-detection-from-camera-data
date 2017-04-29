library(h2o)
library(ggplot2)
h2o.init()

# IMPORT TRAIN FILES

digitsTrainPath = paste(getwd(), "mnist/train.csv", sep = "/")
digitsTrain.hex = h2o.importFile(path = digitsTrainPath)

# IMPORT TEST FILES

digitsTestPath = paste(getwd(), "mnist/test.csv", sep = "/")
digitsTest.hex = h2o.importFile(path = digitsTestPath)

# The data consists of 784(28^2) columns, 785 is a label(0,1,2,...,9)

predictors = c(1:784)
resp = 785

# We do unsupervised learning so we can drop the label column

train.hex <- digitsTrain.hex[,-resp]
test.hex <- digitsTest.hex[,-resp]

# Traing unsupervised deep learning autoencoder

model = h2o.deeplearning(
  x=predictors,
  training_frame = train.hex,
  hidden = c(50),
  epochs = 1,
  activation = "Tanh",
  autoencoder = TRUE,
  ignore_const_cols = FALSE
)

# Find anomalies

anomalies = h2o.anomaly(
  model, 
  test.hex
)
error <- as.data.frame(anomalies)

# Reconstruct original data from autoencoder

predict = h2o.predict(model, test.hex)

# Find data with biggest and smallest reconstruction error

biggest_errors <- order(error[,1], decreasing = FALSE)[c(27991:28000)]
smallest_errors <- order(error[,1], decreasing = FALSE)[c(0:10)]
  
# Visualize data with biggest and smallest reconstruction error

par( mfrow = c(10,10), mai = c(0,0,0,0))
for(i in smallest_errors){
  y = as.matrix(test.hex[i, 1:784])
  dim(y) = c(28, 28)
  image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
  
  y = as.matrix(predict[i, 1:784])
  dim(y) = c(28, 28)
  image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
}

for(i in biggest_errors){
  y = as.matrix(test.hex[i, 1:784])
  dim(y) = c(28, 28)
  image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
  
  y = as.matrix(predict[i, 1:784])
  dim(y) = c(28, 28)
  image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
}

#h2o.shutdown(prompt = TRUE)
