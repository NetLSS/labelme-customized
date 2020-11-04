# -*- encoding: utf-8 -*-

from qtpy.QtCore import Qt
from qtpy import QtWidgets
import qtpy as qt

import labelme.AtiConf as ati

class FlagWidget(QtWidgets.QListWidget):
    last_default_item = None

    def mouseDoubleClickEvent(self, *args, **kwargs):  # sslee
        if self.currentItem() is None:
            return

        if self.last_default_item == self.currentItem():
            self.currentItem().setBackground(Qt.white)
            self.currentItem().setCheckState(Qt.Unchecked)
            self.last_default_item = None
            return

        if self.last_default_item is not None:
            self.last_default_item.setBackground(Qt.white)
        self.currentItem().setBackground(ati.default_flag_color)
        self.currentItem().setCheckState(Qt.Checked)
        self.last_default_item = self.currentItem()

        if ati.is_show_flag_default_message:
            mb = QtWidgets.QMessageBox
            msg = self.tr(
                "Default flag set."
            )
            answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes)
