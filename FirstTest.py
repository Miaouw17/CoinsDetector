from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("model_ex-001_acc-0.403509.h5")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
