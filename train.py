
from opt.base import BaseOpt

from models.test_classifier import Classifier


opt = BaseOpt().get_args()

model = Classifier(opt)