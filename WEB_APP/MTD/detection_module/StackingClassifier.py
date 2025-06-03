import stacking_esemble_classifier as onego


class StackingClassifier():

    def __init__(self, base_classifiers, combiner):  # base_classifiers2
        self.stacking = onego.StackingClassifier(base_classifiers, combiner)  # base_classifiers2

    def fit(self, x, y):
        self.stacking.fit(x, y)

    def predict(self, x):
        return self.stacking.meta_train(x)

    def partial_fit(self, x, y):
        self.stacking.partial_predict(x, y)
