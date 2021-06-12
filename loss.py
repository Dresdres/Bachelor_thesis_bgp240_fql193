import torch
import torch.nn as nn

from utils import intersection_over_union

class YOLOloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_sqrt_err = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 2
        self.lambda_noObject = 5
        self.lambda_object = 5
        self.lambda_box = 8


    def forward(self, predictions, target, anchors):
        obj = target[...,0] == 1
        noObj = target[...,0] == 0


        # No object loss
        noObjLoss = self.bce((predictions[..., 0:1][noObj]), (target[..., 0:1][noObj]) ,)
        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_pred = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5])*anchors], dim=-1)
        ious = intersection_over_union(box_pred[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mean_sqrt_err(self.sigmoid(predictions[..., 0:1][obj]), ious*target[..., 0:1][obj])

        #box coordinate loss
        predictions[...,1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-16+target[..., 3:5] / anchors)
        box_loss = self.mean_sqrt_err(predictions[...,1:5][obj], target[...,1:5][obj])

        #class loss
        class_loss =self.entropy((predictions[...,5:][obj]), (target[...,5][obj].long()),)

        return (
            self.lambda_box*box_loss + self.lambda_object*object_loss
            + self.lambda_noObject*noObjLoss + self.lambda_class*class_loss
        )