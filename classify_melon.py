# Torch Libraries
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

# Others
import numpy as np
import time
from PIL import Image


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MelonClassifier(object):

    def __init__(self) -> None:
        self.MODEL_NAME = "melon_ml/trained_models/melon_squeezenet_data3_20211104-212820.pth"
        self.model, self.input_size = self.initialize_model("squeezenet", 2, feature_extract=True)

        self.data_transforms =  transforms.Compose([
        transforms.Resize(self.input_size),
        transforms.CenterCrop(self.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
            input_size = 224

        elif model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        model_ft.load_state_dict(torch.load(self.MODEL_NAME, map_location=torch.device('cpu')))

        # Put model in eval mode
        model_ft.eval()
        
        return model_ft, input_size



    # Input: Bytes.IO Image
    # Output: boolean True or False
    def classify(self, img):

        class_names = ("melon", "not melon")

        img = Image.open(img).convert("RGB")

        t_image = self.data_transforms(img)

        with torch.no_grad(): 
            t_image = t_image.to(device)
            outputs = self.model(t_image[None, ...])
            _, preds = torch.max(outputs, 1)


        # Class label, Confidence
        label, confidence = [class_names[x] for x in preds][0], outputs[0][0]

        # Print label and confidence
        # print(label)
        # print(confidence)

        if label == "melon" and confidence > 1.0:
            return True
        else:
            return False


def time_classify(melon, img_path):

    from PIL import Image

    img_path = "test_cases/{}.jpeg".format
    img = Image.open(img_path(img_path)).convert("RGB")

    start = time.time()

    pred, confidence = melon.classify(img)
    
    print(f"actual:{img_path}\npred: {pred}\nconfidence: {confidence}\ntime: {time.time()-start}\n")


if __name__ == "__main__":
    pass
    # Example run

    # melon = Classifier()

    # time_classify(melon, img_path)
    





    