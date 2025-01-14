import unittest
import torch
from tensorfusionvlm.constants import EnumTokenType
import random


class TestConst(unittest.TestCase):
    def test_is_ctl(self):
        _values = [
            EnumTokenType.PAD.value,
            EnumTokenType.INST.value,
            EnumTokenType.SYS.value,
            EnumTokenType.IMG.value,
            EnumTokenType.QUE.value,
            EnumTokenType.ANS.value,
            EnumTokenType.BOS.value,
            EnumTokenType.EOS.value,
        ]

        tensor = []
        for _ in range(3):
            tensor.append([random.choice(_values) for _ in range(8)])

        tensor = torch.IntTensor(tensor)

        is_ctl_mask = EnumTokenType.is_ctl(tensor)

        actural = torch.logical_or(
            tensor == EnumTokenType.PAD.value,
            tensor == EnumTokenType.INST.value,
        )

        actural = torch.logical_or(
            actural,
            tensor == EnumTokenType.SYS.value,
        )

        actural = torch.logical_or(
            actural,
            tensor == EnumTokenType.BOS.value,
        )

        actural = torch.logical_or(
            actural,
            tensor == EnumTokenType.EOS.value,
        )

        not_ctl_mask = EnumTokenType.not_ctl(tensor)

        self.assertTrue(torch.all(is_ctl_mask == actural))
        self.assertTrue(torch.all(not_ctl_mask != actural))
