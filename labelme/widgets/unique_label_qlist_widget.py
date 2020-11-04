# -*- encoding: utf-8 -*-

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from .escapable_qlist_widget import EscapableQListWidget

import labelme.AtiConf as ati

class UniqueLabelQListWidget(EscapableQListWidget):
    last_default_item = None

    def mouseDoubleClickEvent(self, *args, **kwargs):  # sslee
        #super(UniqueLabelQListWidget, self).mouseDoubleClickEvent(args, kwargs)

        if self.last_default_item == self.currentItem():
            self.currentItem().setBackground(Qt.white)
            self.last_default_item = None
            return

        if self.last_default_item is not None:
            self.last_default_item.setBackground(Qt.white)
        self.currentItem().setBackground(ati.default_flag_color)
        self.last_default_item = self.currentItem()

        if ati.is_show_label_default_message:
            mb = QtWidgets.QMessageBox
            msg = self.tr(
                "Default label set."
            )
            answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes)

    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemsByLabel(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                items.append(item)
        return items

    def createItemFromLabel(self, label):
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText("{}".format(label))
        else:
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    label, *color
                )
            )
            item.setToolTip(label)  # sslee
        qlabel.setAlignment(Qt.AlignBottom)

        item.setSizeHint(qlabel.sizeHint())

        self.setItemWidget(item, qlabel)
