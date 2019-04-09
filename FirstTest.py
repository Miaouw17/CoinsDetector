from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("img/train/models/model_ex-001_acc-0.403509.h5")
prediction.setJsonPath("img/train/json/model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("img/train/train/2f/1-9.jpeg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
