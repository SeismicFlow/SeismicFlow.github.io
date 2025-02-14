"""
SeismicFlow
Copyright (C) 2025 Matin Mahzad

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os.path
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QScrollArea
from PyQt5 import QtWidgets
from PyQt5.QtGui import QMouseEvent
from matplotlib.colors import LinearSegmentedColormap
import re
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QTabWidget, QSizePolicy, QTabBar
from PyQt5.QtGui import QIcon, QFont, QPixmap
from OpenGL.GLUT import *
from scipy.interpolate import interp1d
from pyqtgraph.opengl import GLVolumeItem
import numpy as np
from PyQt5.QtWidgets import QProgressBar, QSplitter
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLTextItem
from PyQt5.QtGui import QColor
import json
from PyQt5.QtCore import QTimer, QTime
from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QColorDialog, QTimeEdit, QDialog
from PyQt5.QtWidgets import QDialogButtonBox, QListWidget, QSpinBox, QComboBox
import psutil
from PyQt5.QtGui import QVector3D
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
import pyqtgraph as pg
from scipy import ndimage
from pyqtgraph.GraphicsScene import exportDialog
from pyqtgraph import ColorBarItem
from scipy.signal import hilbert, argrelextrema
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableView, QAbstractItemView
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PyQt5.QtWidgets import QInputDialog
import matplotlib.pyplot as plt
import matplotlib
import segyio
import time
import pandas as pd
import os
import torch
import multiprocessing
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr
from PyQt5.QtWidgets import QSpacerItem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import weight_norm
from pytorch_tabnet.tab_model import TabNetRegressor
from ngboost import NGBRegressor
from catboost import CatBoostRegressor
from PyQt5.QtGui import QDoubleValidator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.random_projection import GaussianRandomProjection
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Flatten, Reshape, MultiHeadAttention, LayerNormalization, Add
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, Birch, MeanShift, AffinityPropagation, OPTICS
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
import hdbscan
from minisom import MiniSom
import joblib

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


class CustomGLViewWidget(GLViewWidget):
    def __init__(self, tensor_visualizer, parent=None):
        super().__init__(parent)
        self.tensor_visualizer = tensor_visualizer
        self.background_color_loaded = None
        # Set the initial background color to white (or any color you prefer)
        self.setBackgroundColor(self.load_background_color())
        self.selected_items = set()  # Set to store selected items
        self.opts['distance'] = 35

        # Create a grid item
        self.grid = GLGridItem()
        self.grid.setSize(x=20, y=20, z=20)  # Customize the size if needed
        self.grid.setSpacing(x=1, y=1, z=1)
        self.grid.setColor(self.tensor_visualizer.last_grid_color)
        self.addItem(self.grid)  # Add the grid to the GLViewWidget

        # Calculate the positions for the labels based on the grid size
        label_positions_x = np.linspace(-20 / 2, 20 / 2, 10 + 1)
        label_positions_y = np.linspace(-20 / 2, 20 / 2,
                                        10 + 1)  # Create a QFont object with the desired font size
        # Create a QFont object with the desired font size for the axis labels
        axis_font = QFont()
        axis_font.setPointSize(14)  # Set the font size to 14 points for the axis labels

        # Calculate the middle position for the axis labels
        middle_x = (label_positions_x[0] + label_positions_x[-1]) / 2
        middle_y = (label_positions_y[0] + label_positions_y[-1]) / 2
        offset = 2  # Increased offset

        # Add X-Axis labels with increased offset and specified font size
        self.x_axis_label_bottom = GLTextItem(pos=(middle_x, -20 / 2 - offset, 0), text="X-Axis",
                                              font=axis_font)
        self.x_axis_label_bottom.setData(color=self.tensor_visualizer.last_grid_color)
        self.addItem(self.x_axis_label_bottom)

        self.x_axis_label_top = GLTextItem(pos=(middle_x, 20 / 2 + offset, 0), text="X-Axis",
                                           font=axis_font)
        self.x_axis_label_top.setData(color=self.tensor_visualizer.last_grid_color)
        self.addItem(self.x_axis_label_top)

        # Add Y-Axis labels with increased offset and specified font size
        self.y_axis_label_left = GLTextItem(pos=(-20 / 2 - offset, middle_y, 0), text="Y-Axis",
                                            font=axis_font)
        self.y_axis_label_left.setData(color=self.tensor_visualizer.last_grid_color)
        self.addItem(self.y_axis_label_left)

        self.y_axis_label_right = GLTextItem(pos=(20 / 2 + offset, middle_y, 0), text="Y-Axis",
                                             font=axis_font)
        self.y_axis_label_right.setData(color=self.tensor_visualizer.last_grid_color)
        self.addItem(self.y_axis_label_right)

        if self.tensor_visualizer.grid_active:
            self.grid.show()
            self.x_axis_label_bottom.show()
            self.x_axis_label_top.show()
            self.y_axis_label_left.show()
            self.y_axis_label_right.show()

        else:
            self.grid.hide()
            self.x_axis_label_bottom.hide()
            self.x_axis_label_top.hide()
            self.y_axis_label_left.hide()
            self.y_axis_label_right.hide()

    def save_background_color(self, color):
        # Save the background color to a JSON file
        settings = {'background_color': color.name()}
        with open('settings.json', 'w') as file:
            json.dump(settings, file)

    def load_background_color(self):
        # Try to load the background color from a JSON file
        try:
            with open('settings.json', 'r') as file:
                settings = json.load(file)
                color = QColor(settings['background_color'])
                self.background_color_loaded = color
                return color
        except (FileNotFoundError, KeyError):
            self.background_color_loaded = QColor(99, 99, 99)
            # If the file doesn't exist or the key is not found, use the default color
            return QColor(99, 99, 99)

    def mousePressEvent(self, event):
        # Check if the right mouse button was pressed
        if event.button() == Qt.RightButton:
            # Set the cursor to an arrow
            self.setCursor(QCursor(Qt.ArrowCursor))
        else:
            # For other mouse buttons, change the cursor to a closed hand
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        # Call the parent class's mousePressEvent to ensure default behavior
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Check if the right mouse button was released
        if event.button() == Qt.RightButton:
            # Keep the cursor as an arrow
            self.setCursor(QCursor(Qt.ArrowCursor))
        else:
            # For other mouse buttons, change the cursor to an open hand
            self.setCursor(QCursor(Qt.OpenHandCursor))
        # Call the parent class's mouseReleaseEvent to ensure default behavior
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        change_color_action = context_menu.addAction("Change Background Color")
        switch_to_2d_action = context_menu.addAction("Switch to 2D")

        # Add toggle actions to Centre Data
        Reset_View_action = QAction("Reset View", self)
        context_menu.addAction(Reset_View_action)

        # Add toggle actions for grid and axes
        toggle_grid_action = QAction("Toggle Grid", self)
        context_menu.addAction(toggle_grid_action)

        # Add a new submenu for hiding items
        self.hide_item_submenu = QMenu("Hide Item", self)
        context_menu.addMenu(self.hide_item_submenu)

        # Add a new submenu for removing items
        self.remove_item_submenu = QMenu("Remove Item", self)
        context_menu.addMenu(self.remove_item_submenu)

        # Add toggle actions for grid and axes
        grid_color = QAction("Change Grid Color", self)
        context_menu.addAction(grid_color)

        # Add a new submenu for camera views
        camera_view_submenu = QMenu("Camera View", self)
        context_menu.addMenu(camera_view_submenu)

        # Add Save actions
        Save_Image = QAction("Save Image", self)
        Save_Image.triggered.connect(self.save_image)
        context_menu.addAction(Save_Image)

        # Define camera view actions
        top_view_action = QAction("Top", self)
        bottom_view_action = QAction("Bottom", self)
        left_view_action = QAction("Left", self)
        right_view_action = QAction("Right", self)
        front_view_action = QAction("Front", self)
        back_view_action = QAction("Back", self)

        # Add actions to the submenu
        camera_view_submenu.addAction(top_view_action)
        camera_view_submenu.addAction(bottom_view_action)
        camera_view_submenu.addAction(left_view_action)
        camera_view_submenu.addAction(right_view_action)
        camera_view_submenu.addAction(front_view_action)
        camera_view_submenu.addAction(back_view_action)

        # Connect actions to methods
        top_view_action.triggered.connect(lambda: self.set_camera_view('top'))
        bottom_view_action.triggered.connect(lambda: self.set_camera_view('bottom'))
        left_view_action.triggered.connect(lambda: self.set_camera_view('left'))
        right_view_action.triggered.connect(lambda: self.set_camera_view('right'))
        front_view_action.triggered.connect(lambda: self.set_camera_view('front'))
        back_view_action.triggered.connect(lambda: self.set_camera_view('back'))

        # Call the function to populate the submenus
        self.populate_hide_item_submenu(self.hide_item_submenu)
        self.populate_remove_item_submenu(self.remove_item_submenu)

        action = context_menu.exec_(self.mapToGlobal(event.pos()))

        # Handle actions
        if action == change_color_action:
            color = QColorDialog.getColor()
            if color.isValid():
                self.setBackgroundColor(color)
                grid_clr = self.tensor_visualizer.adjust_grid_color(color)

                self.grid.setColor(grid_clr)

                # Update the color of the x-axis label at the bottom
                self.x_axis_label_bottom.setData(color=grid_clr)

                # Update the color of the x-axis label at the top
                self.x_axis_label_top.setData(color=grid_clr)

                # Update the color of the y-axis label on the left
                self.y_axis_label_left.setData(color=grid_clr)

                # Update the color of the y-axis label on the right
                self.y_axis_label_right.setData(color=grid_clr)

                current_index = self.tensor_visualizer.tab_widget.currentIndex()

                self.tensor_visualizer.grid_color[current_index] = grid_clr

                self.tensor_visualizer.set_grid_color()

        elif action == switch_to_2d_action:
            self.tensor_visualizer.toggle_canvas('2D')
        elif action == toggle_grid_action:
            self.toggleGrid()
        elif action == Reset_View_action:
            self.opts['center'] = QVector3D(0, 0, 0)  # Set the center position to (0, 0, 0)
            self.update()
        elif action == grid_color:
            color = QColorDialog.getColor(options=QColorDialog.ShowAlphaChannel)
            if color.isValid():
                self.grid.setColor(color)

                # Update the color of the x-axis label at the bottom
                self.x_axis_label_bottom.setData(color=color)

                # Update the color of the x-axis label at the top
                self.x_axis_label_top.setData(color=color)

                # Update the color of the y-axis label on the left
                self.y_axis_label_left.setData(color=color)

                # Update the color of the y-axis label on the right
                self.y_axis_label_right.setData(color=color)

                current_index = self.tensor_visualizer.tab_widget.currentIndex()

                self.tensor_visualizer.grid_color[current_index] = color

                self.tensor_visualizer.set_grid_color()

    def save_image(self):
        # Get the index of the currently active tab
        active_tab_index = self.tensor_visualizer.tab_widget.currentIndex()

        # Retrieve the widget of the currently active tab
        active_tab_widget = self.tensor_visualizer.tab_widget.widget(active_tab_index)

        # Assuming the canvas is the first widget in the layout of the active tab
        canvas_layout = active_tab_widget.layout()
        canvas = canvas_layout.itemAt(0).widget()
        # Get the primary screen of the application
        screen = QApplication.primaryScreen()

        # Grab the window associated with the canvas
        screenshot = screen.grabWindow(canvas.winId())

        # Get the file path from the user using a file dialog window
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "PNG Files (*.png)")

        # Check if the user selected a file path
        if file_path:
            # Save the high-quality image
            screenshot.save(file_path, "PNG")

    def set_camera_view(self, view):
        self.opts['center'] = QVector3D(0, 0, 0)  # Set the center position to (0, 0, 0)
        # Define the camera parameters for each view
        views = {
            'bottom': {'elevation': -90, 'azimuth': 0},
            'top': {'elevation': 90, 'azimuth': 0},
            'left': {'elevation': 0, 'azimuth': 90},
            'right': {'elevation': 0, 'azimuth': -90},
            'back': {'elevation': 0, 'azimuth': 180},
            'front': {'elevation': 0, 'azimuth': 0}
        }

        # Get the current opts
        current_opts = self.opts

        # Update the elevation and azimuth based on the selected view
        if view in views:
            current_opts['elevation'] = views[view]['elevation']
            current_opts['azimuth'] = views[view]['azimuth']
            self.update()

    def initial_grid_color(self):
        color = self.tensor_visualizer.last_grid_color  # Set your desired color here

        # Set color using setColor for grids_initial
        for current_index in self.tensor_visualizer.grids_initial:
            self.tensor_visualizer.grids_initial[current_index].setColor(color)

        # Set color using setData for the rest of the dictionaries
        for attr_name in ['x_axis_label_bottom_initial', 'x_axis_label_top_initial',
                          'y_axis_label_left_initial', 'y_axis_label_right_initial']:
            attr = getattr(self.tensor_visualizer, attr_name)
            for current_index in attr:
                attr[current_index].setData(color=color)

    def update_all_instances_background(self, color):
        # Iterate over the dictionary and update the background color for each instance
        for widget in self.tensor_visualizer.custom_gl_widgets.values():
            widget.plot_widget.setBackgroundColor(color)
            self.tensor_visualizer.last_selected_color = color
            self.background_color_loaded = color

    def toggle_item_selection(self, item):
        # Toggle item selection
        if item in self.selected_items:
            self.selected_items.remove(item)  # Deselect the item

            item.show()  # Reset the item's appearance

            print('Item shown')
        else:
            self.selected_items.add(item)  # Select the item

            item.hide()

            print('Item hidden')

    def populate_hide_item_submenu(self, submenu):
        # Populate the submenu with actions to hide items
        counter = 1
        for item in self.tensor_visualizer.tab_volume_items.get(self.tensor_visualizer.tab_widget.currentIndex(), []):
            item_action = QAction(f"Item {counter}: {'Shown' if item.visible() else 'Hidden'}", self)
            item_action.triggered.connect(lambda _, i=item: self.toggle_item_selection(i))
            submenu.addAction(item_action)
            counter += 1

    def populate_remove_item_submenu(self, submenu):
        # Populate the submenu with actions to remove items
        counter = 1
        for item in self.tensor_visualizer.tab_volume_items.get(self.tensor_visualizer.tab_widget.currentIndex(), []):
            item_action = QAction(f"Remove Item {counter}", self)
            item_action.triggered.connect(lambda _, i=item: self.remove_item(i))
            submenu.addAction(item_action)
            counter += 1

    def remove_item(self, item):
        # Remove the item from the dictionary
        current_index = self.tensor_visualizer.tab_widget.currentIndex()
        if item in self.tensor_visualizer.tab_volume_items[current_index]:
            self.tensor_visualizer.tab_volume_items[current_index].remove(item)
            self.removeItem(item)
            # Update the submenus to reflect the removal
            self.populate_hide_item_submenu(self.hide_item_submenu)
            self.populate_remove_item_submenu(self.remove_item_submenu)

    def toggleGrid(self):
        # Get the index of the currently active tab
        active_tab_index = self.tensor_visualizer.tab_widget.currentIndex()

        # Check if the grid for the current tab exists in the TensorVisualizer's grids dictionary
        if active_tab_index in self.tensor_visualizer.grids:
            grid = self.tensor_visualizer.grids[active_tab_index]
            if grid.visible():
                grid.hide()
                if active_tab_index in self.tensor_visualizer.grid_labels:
                    for item in self.tensor_visualizer.grid_labels[active_tab_index]:
                        item.hide()  # Assuming 'item' has a .hide() method
                if active_tab_index in self.tensor_visualizer.grid_axis_labels:
                    for item in self.tensor_visualizer.grid_axis_labels[active_tab_index]:
                        item.hide()  # Assuming 'item' has a .hide() method
            else:
                if active_tab_index in self.tensor_visualizer.grid_labels:
                    for item in self.tensor_visualizer.grid_labels[active_tab_index]:
                        item.show()  # Assuming 'item' has a .hide() method
                if active_tab_index in self.tensor_visualizer.grid_axis_labels:
                    for item in self.tensor_visualizer.grid_axis_labels[active_tab_index]:
                        item.show()  # Assuming 'item' has a .hide() method
                grid.show()
        else:
            # Toggle the initial grid (self.grid)
            if self.grid.visible():
                self.grid.hide()
                self.x_axis_label_bottom.hide()
                self.x_axis_label_top.hide()
                self.y_axis_label_left.hide()
                self.y_axis_label_right.hide()
            else:
                self.grid.show()
                self.x_axis_label_bottom.show()
                self.x_axis_label_top.show()
                self.y_axis_label_left.show()
                self.y_axis_label_right.show()


# PyQtGraphCanvas class for the 3D graph canvas
class PyQtGraphCanvas(QWidget):
    def __init__(self, tensor_visualizer, parent=None):
        super().__init__(parent)
        self.plot_widget = CustomGLViewWidget(tensor_visualizer)

        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self.tensor_data = None  # Placeholder for tensor data

    def enterEvent(self, event):
        # Set cursor to open hand when entering the widget
        self.plot_widget.setCursor(QCursor(Qt.OpenHandCursor))
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Revert cursor to arrow when leaving the widget
        self.plot_widget.setCursor(QCursor(Qt.ArrowCursor))
        super().leaveEvent(event)


class ResNet1D(nn.Module):
    def __init__(self, input_dim):
        super(ResNet1D, self).__init__()
        self.model = resnet50(weights=None)  # Initialize without pre-trained weights

        # Modify the first convolutional layer to accept 1D input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the fully connected layer to match the number of input features and output a single value
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        # Additional linear layer to convert flattened 1D input to a suitable shape for conv layer
        self.fc1 = nn.Linear(input_dim, 64 * 7 * 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)  # Apply the linear layer to get the right shape
        x = self.relu(x)
        x = x.view(-1, 1, 56, 56)  # Reshape to (batch_size, channels, height, width)
        return self.model(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2)  # Global average pooling
        return self.fc(x)


class RoundedCanvas(pg.PlotWidget):
    def __init__(self, tensor_visualizer, figure=None):
        super().__init__()
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setMenuEnabled(False)  # This will disable the context menu
        self.background_color_loaded = None
        self.tensor_visualizer = tensor_visualizer
        self.setBackground(self.tensor_visualizer.last_selected_color)

    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)  # Set the cursor to a fist
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)
            super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)


class CustomMessageBox(QMessageBox):
    def __init__(self, theme_is_dark, text, icon, buttons, default_button, parent=None):
        super().__init__(parent)
        self.theme_is_dark = theme_is_dark
        self.setText(text)
        self.setIcon(icon)
        self.setStandardButtons(buttons)
        self.setDefaultButton(default_button)
        self.set_background_color()

    def set_background_color(self):
        if self.theme_is_dark:
            self.setStyleSheet("background-color: #353535; color: Gainsboro;")
        else:
            self.setStyleSheet("background-color: #B2B2B2; color: 353535;")


class CustomColorBarItem(ColorBarItem):
    def __init__(self, *args, **kwargs):
        super(CustomColorBarItem, self).__init__(*args, **kwargs)
        # Disable the default context menu
        self.setMenuEnabled(False)


class TensorVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_custom_cursor()
        self.save_option_state = True  # Initialize the save option state
        self.save_directory = None
        self.save_file_name = None
        self.dpi = None  # Initialize the DPI reference
        self.custom_gl_widgets = {}  # Dictionary to track CustomGLViewWidget instances
        self.loaded_file_paths = {}  # Initialize an empty dictionary for file paths
        self.recent_files_path = 'json.json'
        self.load_recent_files()  # Add this line to call the method on startup
        self.file_path = None
        self.file_name = None
        self.isDarkTheme = False  # Set your theme condition here
        self.update_theme_initial_on_time()
        self.view_type = {0: '2D'}
        self.analytic_signal = {}
        self.envelope_dict = {}
        self.inst_phase_radians = {}
        self.inst_freq_hz_padded = {}
        self.inst_bandwidth_dict = {}
        self.kmeans_dict = {}
        self.num_clusters = {}
        # Initialize the dictionary to store tensors
        self.tensor_dict = {}
        # UI elements
        self.last_grid_color = '#FFFFFF'
        self.grid_color = {}
        # Initialize state variables for Grid and Axes
        self.grid_active = True
        self.cglv = CustomGLViewWidget(self)
        self.last_selected_color = self.cglv.background_color_loaded  # Default color
        self.last_grid_color = self.adjust_grid_color(self.last_selected_color)
        self.r_canvas = RoundedCanvas(self)  # Store the instance as a class attribute
        self.create_widgets()
        self.dark_mode_start_time = QTime(20, 0)  # 20:00 in 24-hour format
        self.dark_mode_end_time = QTime(6, 0)  # 08:00 in 24-hour format
        self.automatic_theme_enabled = True
        # Set up a timer to check the time every minute
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_theme_based_on_time)
        self.timer.start(60000)  # Check every 60 seconds
        self.update_theme_based_on_time()  # Initial theme update
        self.setWindowTitle("Seismic Flow")
        self.setWindowIcon(QIcon('icon.png'))  # Set the icon for the dialog
        self.cbar = {}
        # Placeholder for tensor data
        self.tensor_data = None
        self.interpolated_slice_2d = {}
        # Link the HistogramLUTItem to the ImageItem
        self.img_item = {}
        self.tab_3D_state = {}  # Dictionary to track the 3D plot state for each tab
        self.tab_last_plotted_data_name = {}  # Dictionary to track the file name of the last data plotted on each tab
        self.color_mapp = None
        self.pyqtgraph_canvas = PyQtGraphCanvas(self)  # Pass self as the tensor_visualizer reference
        self.units_dict = {}  # Add this to store units for each file
        self.grids = {}  # Dictionary to store grid instances for each tab
        self.splitter_state = self.splitter.saveState()
        self.grids_initial = {}
        self.well_color_mapping = 'viridis'
        self.x_axis_label_bottom_initial = {}
        self.x_axis_label_top_initial = {}
        self.y_axis_label_left_initial = {}
        self.y_axis_label_right_initial = {}
        # Initialize the dictionary
        self.model_dict = {}

        self.tab_volume_items = {}

        self.canvas = CustomGLViewWidget(self)  # Pass self as a reference to the CustomGLViewWidget

        self.grid_labels = {}
        self.grid_axis_labels = {}

    def init_custom_cursor(self):
        # Load your custom cursor image
        pixmap = QPixmap('busy.png')
        scaled_pixmap = pixmap.scaled(QSize(20, 20), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Create the custom cursor
        self.custom_cursor = QCursor(scaled_pixmap, -1, -1)

    def create_widgets(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Store a reference to central_widget
        self.central_widget = central_widget

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 0)  # Adjust margins as needed

        # Create a splitter to hold the left and right containers
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Create a container for the left section
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        self.statusBar().setFixedHeight(20)

        # Disable the size grip
        self.statusBar().setSizeGripEnabled(False)

        # Create a progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)  # Set the range (0-100%)
        self.progress_bar.setValue(0)  # Initialize to 0%
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.progress_bar.setMaximumWidth(150)  # Adjust the width as desired

        # Create a label for memory usage
        self.memory_label = QLabel()
        self.memory_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.memory_label.setAlignment(Qt.AlignLeft)

        label_widget = QWidget()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 19, 0)
        self.layout.addWidget(self.memory_label)
        label_widget.setLayout(self.layout)
        self.statusBar().addPermanentWidget(label_widget)

        # Update memory usage and responsiveness periodically
        self.update_memory_usage()

        self.menubar = self.menuBar()
        file_menu = QMenu("File", self)

        # Create a submenu for "Recent Files"
        self.recent_files_menu = QMenu("Recent Files", self)
        self.update_recent_files_submenu()  # Populate the submenu initially

        file_menu.addMenu(self.recent_files_menu)  # Add the "Recent Files" submenu

        # Add '3D Canvas Appearance' menu with submenus
        open_menu = QMenu("Open", self)
        file_menu.addMenu(open_menu)

        # Add 'Grid' submenu with state-dependent text
        self.open_action = QAction("Numpy", self)
        self.open_action.triggered.connect(self.load_tensor)
        open_menu.addAction(self.open_action)

        # Add 'Grid' submenu with state-dependent text
        self.opens_action = QAction("Segy", self)
        self.opens_action.triggered.connect(self.process_seismic_data)
        open_menu.addAction(self.opens_action)

        # Add 'Grid' submenu with state-dependent text
        self.load_las_file_action = QAction("Las", self)
        self.load_las_file_action.triggered.connect(self.load_las_file)
        open_menu.addAction(self.load_las_file_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.closeEvent)  # Connect to your file handling function

        file_menu.addSeparator()  # Add another separator
        file_menu.addAction(exit_action)
        # Add the "File" menu to the menu bar
        self.menubar.addMenu(file_menu)

        # Create the 'View' menu
        view_menu = self.menuBar().addMenu("View")

        # Add 'Choose Dark Mode Times' action
        choose_times_action = QAction("Choose Dark Mode Schedule", self)
        choose_times_action.triggered.connect(self.choose_dark_mode_times)
        view_menu.addAction(choose_times_action)

        # Add 'Choose Background Color' action
        choose_bg_color_action = QAction("Choose Background Color", self)
        choose_bg_color_action.triggered.connect(self.choose_background_color)
        view_menu.addAction(choose_bg_color_action)

        # Add 'Choose Menu Bar Color' action
        choose_menu_bar_color_action = QAction("Choose Menu Bar Color", self)
        choose_menu_bar_color_action.triggered.connect(self.choose_menu_bar_color)
        view_menu.addAction(choose_menu_bar_color_action)

        # Add 'Axes' submenu with state-dependent text
        self.canvas_color_action = QAction("Choose Canvas Background Color", self)
        self.canvas_color_action.triggered.connect(self.canvas_color)
        view_menu.addAction(self.canvas_color_action)

        # Add 'Toggle Canvas' action with dynamic text
        self.toggle_canvas_action = QAction("3D View", self)
        view_menu.addAction(self.toggle_canvas_action)

        # Add '3D Canvas Appearance' menu with submenus
        canvas_appearance_menu = QMenu("3D Canvas Appearance", self)
        view_menu.addMenu(canvas_appearance_menu)

        # Add 'Grid' submenu with state-dependent text
        self.grid_action = QAction("Deactivate Grid", self)
        self.grid_action.triggered.connect(self.toggle_grid)
        canvas_appearance_menu.addAction(self.grid_action)

        # Add 'Axes' submenu with state-dependent text
        self.grid_color_action = QAction("Choose Grid Color", self)
        self.grid_color_action.triggered.connect(self.apply_grid_color)
        canvas_appearance_menu.addAction(self.grid_color_action)

        # Add 'Choose Background Color' action
        self.toggle_layout_action = QAction("Hide Controls", self)
        self.toggle_layout_action.triggered.connect(self.toggle_left_layout)
        view_menu.addAction(self.toggle_layout_action)

        # Create the 'View' menu
        tools_menu = self.menuBar().addMenu("Tools")

        # Add '3D Canvas Appearance' menu with submenus
        seismic_attribute_menu = QMenu("Seismic Attributes", self)
        tools_menu.addMenu(seismic_attribute_menu)

        # Add 'Axes' submenu with state-dependent text
        self.fault_prediction_action = QAction("Fault Prediction", self)
        self.fault_prediction_action.triggered.connect(self.fault_prediction)
        tools_menu.addAction(self.fault_prediction_action)

        # Add 'Axes' submenu with state-dependent text
        self.thresholded_amplitude_action = QAction("Amplitude Thresholding", self)
        self.thresholded_amplitude_action.triggered.connect(self.thresholded_amplitude)
        tools_menu.addAction(self.thresholded_amplitude_action)

        # Add 'Axes' submenu with state-dependent text
        self.kmeans_amplitude_action = QAction("Kmeans Amplitude Clustering", self)
        self.kmeans_amplitude_action.triggered.connect(self.kmeans_amplitude)
        tools_menu.addAction(self.kmeans_amplitude_action)

        # Add 'Axes' submenu with state-dependent text
        self.facies_prediction_action = QAction("Facies Prediction", self)
        self.facies_prediction_action.triggered.connect(self.facies_prediction)
        tools_menu.addAction(self.facies_prediction_action)

        # Add 'Axes' submenu with state-dependent text
        self.tensor_cutter_action = QAction("Tensor Trimmer", self)
        self.tensor_cutter_action.triggered.connect(self.tensor_cutter)
        tools_menu.addAction(self.tensor_cutter_action)

        # Add 'Axes' submenu with state-dependent text
        self.open_plot_dialog_action = QAction("Well Analysis", self)
        self.open_plot_dialog_action.triggered.connect(self.open_plot_dialog)
        tools_menu.addAction(self.open_plot_dialog_action)

        # Add 'Axes' submenu with state-dependent text
        self.Instantaneous_Phase_action = QAction("Instantaneous Phase", self)
        self.Instantaneous_Phase_action.triggered.connect(self.instantaneous_phase)
        seismic_attribute_menu.addAction(self.Instantaneous_Phase_action)

        # Add 'Axes' submenu with state-dependent text
        self.Instantaneous_frequency_action = QAction("Instantaneous Frequency", self)
        self.Instantaneous_frequency_action.triggered.connect(self.instantaneous_frequency)
        seismic_attribute_menu.addAction(self.Instantaneous_frequency_action)

        # Add 'Axes' submenu with state-dependent text
        self.inst_bandwidth_action = QAction("Instantaneous Bandwidth", self)
        self.inst_bandwidth_action.triggered.connect(self.inst_bandwidth)
        seismic_attribute_menu.addAction(self.inst_bandwidth_action)

        # Add 'Axes' submenu with state-dependent text
        self.dominant_frequency_action = QAction("Dominant Frequency", self)
        self.dominant_frequency_action.triggered.connect(self.dominant_frequency)
        seismic_attribute_menu.addAction(self.dominant_frequency_action)

        # Add 'Axes' submenu with state-dependent text
        self.cosine_of_phase_action = QAction("Cosine of Phase", self)
        self.cosine_of_phase_action.triggered.connect(self.cosine_of_phase)
        seismic_attribute_menu.addAction(self.cosine_of_phase_action)

        # Add 'Axes' submenu with state-dependent text
        self.envelope_action = QAction("Envelope", self)
        self.envelope_action.triggered.connect(self.envelope)
        seismic_attribute_menu.addAction(self.envelope_action)

        # Add 'Axes' submenu with state-dependent text
        self.sweetness_action = QAction("Sweetness", self)
        self.sweetness_action.triggered.connect(self.sweetness)
        seismic_attribute_menu.addAction(self.sweetness_action)

        # Add 'Axes' submenu with state-dependent text
        self.apparent_polarity_action = QAction("Apparent Polarity", self)
        self.apparent_polarity_action.triggered.connect(self.apparent_polarity)
        seismic_attribute_menu.addAction(self.apparent_polarity_action)

        # Add 'Axes' submenu with state-dependent text
        self.rms_amplitude_action = QAction("Rms Amplitude", self)
        self.rms_amplitude_action.triggered.connect(self.rms_amplitude)
        seismic_attribute_menu.addAction(self.rms_amplitude_action)

        # Create the 'Help' menu
        help_menu = self.menuBar().addMenu("Help")

        # Add 'Choose Dark Mode Times' action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel('Data')
        self.treeWidget.setMaximumHeight(180)  # Adjust the height as needed
        left_layout.addWidget(self.treeWidget)  # Add the tree widget to the layout
        self.populateList()
        self.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget.customContextMenuRequested.connect(self.openMenu)

        left_layout.addSpacing(10)

        # file directory
        self.file_entry = QLineEdit()

        # Placeholder for displaying tensor shape
        self.shape_label = QLabel(alignment=Qt.AlignCenter)

        self.dim1_entry = QLineEdit()
        self.dim2_entry = QLineEdit()
        self.index_dim_entry = QLineEdit()

        # Add buttons for selecting plotting direction
        self.time_slice_button = QPushButton("Time Slice", checkable=True, clicked=self.set_time_slice)
        self.cross_line_button = QPushButton("Cross Line", checkable=True, clicked=self.set_cross_line)
        self.inline_button = QPushButton("Inline", checkable=True, clicked=self.set_inline)
        self.Three_D_button = QPushButton("3D", checkable=True, clicked=self.three_d)

        # Create button group for exclusive behavior
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.time_slice_button)
        self.button_group.addButton(self.cross_line_button)
        self.button_group.addButton(self.inline_button)
        self.button_group.addButton(self.Three_D_button)

        left_layout.addWidget(self.time_slice_button)
        left_layout.addWidget(self.cross_line_button)
        left_layout.addWidget(self.inline_button)
        left_layout.addWidget(self.Three_D_button)
        self.Three_D_button.hide()

        self.index_label = QLabel("Interval:")
        self.index_entry = QLineEdit()
        left_layout.addWidget(self.index_label)
        left_layout.addWidget(self.index_entry)
        # Create a slider for the interval index
        self.index_slider = QSlider(Qt.Horizontal)
        self.index_slider.setTickPosition(QSlider.NoTicks)
        # Connect the textChanged signal of the index entry box to a slot that updates the slider
        self.index_entry.textChanged.connect(self.update_slider_from_entry)

        self.index_label.hide()
        self.index_entry.hide()
        self.index_slider.hide()

        # Add the slider to the UI layout
        left_layout.addWidget(self.index_slider)

        self.sampling_interval_label = QLabel("Sampling Interval:")
        self.sampling_interval_entry = QLineEdit()
        self.sampling_interval_label.hide()
        self.sampling_interval_entry.hide()
        left_layout.addWidget(self.sampling_interval_label)
        left_layout.addWidget(self.sampling_interval_entry)

        # Initially hide the Channel Index entry box
        self.channel_label = QLabel("Channel Dimension:")
        self.channel_entry = QLineEdit()
        self.channel_label.hide()
        self.channel_entry.hide()
        left_layout.addWidget(self.channel_label)
        left_layout.addWidget(self.channel_entry)

        # Initially hide the Channel Index entry box
        self.channel_index_label = QLabel("Channel Index:")
        self.channel_index_entry = QLineEdit()
        self.channel_index_label.hide()
        self.channel_index_entry.hide()
        left_layout.addWidget(self.channel_index_label)
        left_layout.addWidget(self.channel_index_entry)
        # Create a slider for the interval index
        self.Channel_slider = QSlider(Qt.Horizontal)
        self.Channel_slider.setTickPosition(QSlider.NoTicks)
        # Connect the textChanged signal of the index entry box to a slot that updates the slider
        self.channel_index_entry.textChanged.connect(self.update_channel_slider_from_entry)
        self.Channel_slider.valueChanged.connect(self.channel_update)
        self.Channel_slider.hide()

        # Add the slider to the UI layout
        left_layout.addWidget(self.Channel_slider)

        self.plot_button = QPushButton("Plot", clicked=self.plot_tensor)
        left_layout.addWidget(self.plot_button)

        left_layout.addStretch()
        splitter.addWidget(left_container)

        # Create a container for the plot on the right
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)

        # Create a QTabWidget to hold different visualizations
        self.tab_widget = QTabWidget()

        # Create a font with smaller size for the tab labels
        tab_font = QFont()
        tab_font.setPointSize(8)  # Adjust the font size as needed

        # Apply the smaller font to the tab labels
        self.tab_widget.setStyleSheet("QTabBar::tab { font-size: 8pt; }")

        # Create a tab for the rounded canvas
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.rounded_canvas = RoundedCanvas(self)
        self.plot_layout.addWidget(self.rounded_canvas)
        self.tab_widget.addTab(self.plot_tab, f"2D Plot 1 ")

        # Set the closeable tabs
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        # Add the tab widget to the layout
        plot_layout.addWidget(self.tab_widget)

        plot_layout.setContentsMargins(10, 10, 10, 0)

        # Remove the close button from the first tab
        self.tab_widget.tabBar().setTabButton(0, QTabBar.RightSide, None)

        # Add the plus button in front of the last tab label
        self.add_tab_button = QPushButton("+", clicked=self.add_tab)
        self.add_tab_button.setFixedSize(15, 15)  # Set the button size
        self.add_tab_button.setToolTip("Add Tab")  # Add tooltip for the button
        self.add_tab_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.tab_widget.tabBar().setTabButton(self.tab_widget.count() - 1, QTabBar.RightSide, self.add_tab_button)
        self.tab_widget.currentChanged.connect(self.on_tab_change)
        splitter.addWidget(plot_container)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        self.splitter = splitter  # Store a reference to the splitter

        self.rounded_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rounded_canvas.customContextMenuRequested.connect(self.show_context_menu2)
        self.toggle_canvas_action.triggered.connect(self.toggle_canvas_from_menu)

    def toggle_left_layout(self):
        if self.splitter.sizes()[0] == 0:
            # Restore the splitter state to the previously saved state
            self.splitter.restoreState(self.splitter_state)
            self.toggle_layout_action.setText("Hide Controls")
        else:
            # Save the current splitter state before hiding the left layout
            self.splitter_state = self.splitter.saveState()
            # Collapse the left layout by setting its width to 0
            self.splitter.setSizes([0, self.splitter.sizes()[1]])
            self.toggle_layout_action.setText("Show Controls")

    def channel_update(self, value):
        self.channel_index_entry.setText(str(value))

    def update_channel_slider_from_entry(self):
        try:
            # Get the value from the index entry box
            value = int(self.channel_index_entry.text())

            # Get the maximum value allowed for the slider
            max_value = self.Channel_slider.maximum()

            # Check if the entered value exceeds the maximum limit
            if value > max_value:
                # Set the value to the maximum limit
                value = max_value

                # Update the text in the entry box
                self.channel_index_entry.setText(str(value))

            # Set the slider value
            self.Channel_slider.setValue(value)
        except ValueError:
            pass  # Handle the case where the text in the entry box is not a valid integer

    def set_slider_range(self, max_value):
        self.Channel_slider.setMinimum(0)
        self.Channel_slider.setMaximum(max_value - 1)  # Adjusted to match zero-based indexing

    def about(self):
        dialog = QDialog()
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle('About Seismic Flow')
        dialog.setWindowIcon(QIcon('icon.png'))  # Set the icon for the dialog

        # Main layout is horizontal, with two sections
        mainLayout = QHBoxLayout()

        # Left layout for the image
        leftLayout = QVBoxLayout()
        imageLabel = QLabel()
        pixmap = QPixmap('icon.png')  # Load the image
        imageLabel.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        leftLayout.addWidget(imageLabel)
        leftLayout.setAlignment(Qt.AlignTop)
        leftLayout.setContentsMargins(0, 0, 5, 5)  # Adjust margins as needed
        # Right layout for the text
        rightLayout = QVBoxLayout()

        # Title label with larger font and bold
        titleLabel = QLabel('Seismic Flow')
        titleFont = QFont('Arial', 18, QFont.Bold)
        titleLabel.setFont(titleFont)
        titleLabel.setAlignment(Qt.AlignCenter)
        rightLayout.addWidget(titleLabel)

        # Version and developer info with a smaller font
        infoFont = QFont('Arial', 10)
        versionLabel = QLabel('Version 1.0')
        versionLabel.setFont(infoFont)
        rightLayout.addWidget(versionLabel)

        developerLabel = QLabel('Developed by Dr. Matin Mahzad')
        developerLabel.setFont(infoFont)
        rightLayout.addWidget(developerLabel)

        emailLabel = QLabel('Contact: matinmahzad@yahoo.com')
        emailLabel.setFont(infoFont)
        rightLayout.addWidget(emailLabel)

        # Add copyright notice with italic font
        copyrightLabel = QLabel('© 2024 Matin Mahzad. All rights reserved.')
        copyrightFont = QFont('Arial', 8, QFont.Normal, italic=True)
        copyrightLabel.setFont(copyrightFont)
        rightLayout.addWidget(copyrightLabel)

        # Add layouts to the main layout
        mainLayout.addLayout(leftLayout)
        mainLayout.addLayout(rightLayout)

        # Set dialog layout
        dialog.setLayout(mainLayout)

        # Define the styles for light and dark themes
        light_theme_style = """
        QDialog {
            background-color: #B2B2B2; /* Light grey background */
            color: #333333; /* Dark grey text */
        }
        /* Add other widget styles for light theme here */
        """
        # Define the dark theme style
        dark_theme_style = """
        QWidget {
        background-color: #333333; /* Dark grey background */
        color: #FFFFFF; /* White text for better readability */
        }
        /* Add other widget styles for dark theme here */
        """

        # Apply the appropriate style based on the theme
        if self.isDarkTheme:
            dialog.setStyleSheet(dark_theme_style)
        else:
            dialog.setStyleSheet(light_theme_style)

        # Show the dialog
        dialog.exec_()

    def openMenu(self, position):
        item = self.treeWidget.itemAt(position)

        if item is not None:
            # Create the menu
            menu = QMenu()

            # Define the styles for light and dark themes
            light_theme_style = """
            QMenu {
            background-color: #F0F0F0; /* Light grey background */
            color: #333333; /* Dark grey text */
            }
            QMenu::item {
            background-color: #F0F0F0;
            color: #333333;
            }
            QMenu::item:selected {
            background-color: #B2B2B2; /* Slightly darker grey for selected item */
            color: #333333;
            }
            """
            dark_theme_style = """
            QMenu {
            background-color: #404040; /* Dark grey background */
            color: #FFFFFF; /* White text */
            }
            QMenu::item {
            background-color: #404040;
            color: #FFFFFF;
            }
            QMenu::item:selected {
            background-color: #5C5C5C; /* Slightly lighter grey for selected item */
            color: #FFFFFF;
            }
            """

            # Apply the appropriate style based on the theme
            if self.isDarkTheme:
                menu.setStyleSheet(dark_theme_style)
            else:
                menu.setStyleSheet(light_theme_style)

            # Add actions
            if self.file_path.endswith('.npy'):
                menu.addAction('Statistics', self.analysis)
                menu.addAction('Save', self.save_tensor)
            else:
                menu.addAction('Show CSV', self.show_csv_in_dialog)
                menu.addAction('Save', self.save_tensor)
            menu.exec_(self.treeWidget.viewport().mapToGlobal(position))

    def save_tensor(self):
        if self.file_path.endswith('.npy'):
            # Placeholder for future functionality
            selected_items = self.treeWidget.selectedItems()
            if selected_items:
                selected_item = selected_items[0]
                parent_item = selected_item.parent()
                is_parent = parent_item is None

                # Determine the key based on whether the selected item is a parent or a child
                key = selected_item.text(
                    0).strip() if is_parent else f"{parent_item.text(0).strip()}_{selected_item.text(0).strip()}"

                if is_parent:
                    parent_key = selected_item.text(0).strip()
                else:
                    parent_key = parent_item.text(0).strip()

                data = self.tensor_dict[key]
                # Get the file path from the user using a file dialog window
                file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "NumPy Files (*.npy)")

                # Check if the user selected a file path
                if file_path:
                    # Append extra information before the extension
                    info = os.path.basename(self.loaded_file_paths[parent_key])
                    parts = info.split("_")
                    # Find the index of the first occurrence of "tensor"
                    index = parts.index("tensor")
                    # Reconstruct the desired portion of the base name
                    extra_info = "_" + "_".join(parts[index:])[:-len(".npy")]
                    file_dir, file_name = os.path.split(file_path)
                    base_name, extension = os.path.splitext(file_name)
                    new_file_name = f"{base_name}{extra_info}{extension}"
                    new_file_path = os.path.join(file_dir, new_file_name)

                    # Save the NumPy tensor to the modified file path
                    np.save(new_file_path, data)
        else:
            selected_items = self.treeWidget.selectedItems()
            if selected_items:
                selected_item = selected_items[0]
                parent_item = selected_item.parent()
                is_parent = parent_item is None

                # Determine the key based on whether the selected item is a parent or a child
                key = selected_item.text(
                    0).strip() if is_parent else f"{parent_item.text(0).strip()}_{selected_item.text(0).strip()}"

                if is_parent:
                    parent_key = selected_item.text(0).strip()
                else:
                    parent_key = parent_item.text(0).strip()

                data = self.tensor_dict[key]
                # Open a QFileDialog for the user to enter the CSV file path
                csv_file_path, _ = QFileDialog.getSaveFileName(self, 'Save CSV', '', 'CSV Files (*.csv)')

                if csv_file_path:
                    # Save the DataFrame to the specified path
                    data.to_csv(csv_file_path, index=False)

    def analysis(self):
        # Placeholder for future functionality
        selected_items = self.treeWidget.selectedItems()
        if selected_items:
            QApplication.setOverrideCursor(self.custom_cursor)
            selected_item = selected_items[0]
            parent_item = selected_item.parent()
            is_parent = parent_item is None

            # Determine the key based on whether the selected item is a parent or a child
            key = selected_item.text(
                0).strip() if is_parent else f"{parent_item.text(0).strip()} {selected_item.text(0).strip()}"

            data = self.tensor_dict[key]
            # Open the dialog window with the data analysis
            self.openDataAnalysisDialog(data, key)

    def openDataAnalysisDialog(self, data, key):
        dialog = QDialog()
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowIcon(QIcon('icon.png'))  # Set the icon for the dialog
        dialog.setWindowTitle("Statistics")
        mainLayout = QHBoxLayout(dialog)  # Set layout on dialog

        self.leftLayout = QVBoxLayout()

        self.statsArea = QScrollArea()
        self.statsArea.setWidgetResizable(True)
        self.leftLayout.addWidget(self.statsArea)

        self.rightLayout = QVBoxLayout()
        self.histogramView = RoundedCanvas(self)
        self.rightLayout.addWidget(self.histogramView)
        self.histogramView.showAxis('bottom')
        self.histogramView.showAxis('left')
        # Parse the key to determine the color mapping
        parts = key.split()
        if len(parts) == 1:
            title = 'seismic Amplitude'
        else:
            title = ' '.join(parts[1:]).lower().replace(' ', ' ')
        self.histogramView.setLabel('bottom', title)
        self.histogramView.setLabel('left', 'Count')

        mainLayout.addLayout(self.leftLayout, 2)
        mainLayout.addLayout(self.rightLayout, 4)

        self.statsLayout = QVBoxLayout()  # Initialize statsLayout here
        statsWidget = QWidget()
        statsWidget.setLayout(self.statsLayout)
        self.statsArea.setWidget(statsWidget)
        # Analyze the tensor and update the statsLayout and histogramView
        self.analyze_tensor(data, self.statsLayout)

        # Define the styles for light and dark themes
        light_theme_style = """
        QDialog {
            background-color: #B2B2B2; /* Light grey background */
            color: #333333; /* Dark grey text */
        }
        /* Add other widget styles for light theme here */
        """
        # Define the dark theme style
        dark_theme_style = """
        QWidget {
        background-color: #333333; /* Dark grey background */
        color: #FFFFFF; /* White text for better readability */
        }
        /* Add other widget styles for dark theme here */
        """

        # Apply the appropriate style based on the theme
        if self.isDarkTheme:
            dialog.setStyleSheet(dark_theme_style)
        else:
            dialog.setStyleSheet(light_theme_style)

        dialog.exec_()

    def analyze_tensor(self, tensor, layout):
        layout.addWidget(QLabel(f"Tensor shape: {tensor.shape}"))
        layout.addWidget(QLabel(f"Number of samples in total : {tensor.shape[0] * tensor.shape[1] * tensor.shape[2]}"))
        layout.addWidget(QLabel(f"Number of samples per trace : {tensor.shape[0]}"))
        layout.addWidget(QLabel(f"Number of inlines : {tensor.shape[1]}"))
        layout.addWidget(QLabel(f"Number of crosslines : {tensor.shape[2]}"))
        layout.addWidget(QLabel(f"Sampling interval  : {self.get_sampling_interval_from_file_name()}"))
        filename = os.path.basename(self.file_path)
        # Extract the min and max values from the filename
        parts = filename.split('_tensor_')[-1].split('_')
        x_min, x_max, y_min, y_max = map(float,
                                         parts[:4])  # Only take the first four values after 'tensor_'
        layout.addWidget(QLabel(f"Minimum X : {x_min}"))
        layout.addWidget(QLabel(f"Maximum X : {x_max}"))
        layout.addWidget(QLabel(f"Minimum Y : {y_min}"))
        layout.addWidget(QLabel(f"Maximum Y  : {y_max}"))
        layout.addWidget(QLabel(f"inline length : {x_max - x_min}"))
        layout.addWidget(QLabel(f"inline spacing : {(x_max - x_min) / tensor.shape[2]}"))
        layout.addWidget(QLabel(f"Crossline length : {y_max - y_min}"))
        layout.addWidget(QLabel(f"Crossline spacing : {(y_max - y_min) / tensor.shape[1]}"))
        channel_dimension = np.argmin(tensor.shape)

        num_channels = tensor.shape[channel_dimension]
        for i in range(num_channels):
            channel_data = tensor.take(i, axis=channel_dimension)

            mean = np.mean(channel_data)
            std_dev = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            zero_count = np.size(channel_data) - np.count_nonzero(channel_data)
            positive_count = np.sum(channel_data > 0)
            negative_count = np.sum(channel_data < 0)
            unique_numbers = np.unique(channel_data).size

            max_pos = np.unravel_index(np.argmax(channel_data), channel_data.shape)
            min_pos = np.unravel_index(np.argmin(channel_data), channel_data.shape)

            layout.addWidget(QLabel(f"Mean: {mean}"))
            layout.addWidget(QLabel(f"Standard Deviation: {std_dev}"))
            layout.addWidget(QLabel(f"Minimum Value: {min_val} at position {min_pos}"))
            layout.addWidget(QLabel(f"Maximum Value: {max_val} at position {max_pos}"))
            layout.addWidget(QLabel(f"Number of Zeros: {zero_count}"))
            layout.addWidget(QLabel(f"Number of Positive Values: {positive_count}"))
            layout.addWidget(QLabel(f"Number of Negative Values: {negative_count}"))
            layout.addWidget(QLabel(f"Number of Unique Numbers: {unique_numbers}"))

            hist_data, bins = np.histogram(channel_data, bins='auto')
            bins = np.linspace(channel_data.min(), channel_data.max(), len(hist_data) + 1)  # Adjust bins
            self.histogramView.plot(bins, hist_data, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        QApplication.restoreOverrideCursor()

    def adjust_grid_color(self, last_selected_color):
        # Convert the last selected color to a QColor object
        background_color = QColor(last_selected_color)

        # Calculate the luminance of the background color
        luminance = (
                            0.299 * background_color.red() + 0.587 * background_color.green() + 0.114 * background_color.blue()) / 255

        # If the background color is light, set the grid color to black with transparency, otherwise set it to white with transparency
        grid_color = QColor(0, 0, 0, 64) if luminance > 0.5 else QColor(255, 255, 255, 64)

        # Return the grid color as a string in the format '#RRGGBBAA'
        return grid_color

    def populateList(self):
        self.treeWidget.clear()

        # Create a mapping of parent data keys to their tree items
        parent_items = {}

        for key in self.tensor_dict.keys():
            # Assume the key format is 'dataName_attributeName'
            parts = key.split('_', 1)
            data_name = parts[0]
            attribute_name = parts[1] if len(parts) > 1 else ""

            # Find or create the parent item
            if data_name not in parent_items:
                parent_item = QTreeWidgetItem(self.treeWidget, [data_name])
                parent_items[data_name] = parent_item
            else:
                parent_item = parent_items[data_name]

            # Create a child item for the attribute if it exists
            if attribute_name:
                QTreeWidgetItem(parent_item, [attribute_name])

        # Expand all items to make them visible
        self.treeWidget.expandAll()

        # Connect the item selection change to the handleSelectionChange function
        self.treeWidget.itemSelectionChanged.connect(self.handleSelectionChange)

    def handleSelectionChange(self):
        selected_items = self.treeWidget.selectedItems()
        if not selected_items:
            return

        # Assuming single selection, we use the first selected item
        selected_item = selected_items[0]
        key = self.getKeyFromItem(selected_item)
        if key:
            self.selectTensor(key)

    def getKeyFromItem(self, item):
        # Determine if the item is a parent or child and return the appropriate key
        parent_item = item.parent()
        if parent_item is None:
            # This is a parent item
            key = item.text(0).strip()
        else:
            # This is a child item
            parent_key = parent_item.text(0).strip()
            child_key = item.text(0).strip()
            key = f"{parent_key}_{child_key}"
        return key

    def selectTensor(self, key):
        # Remove the state parameter as it's no longer needed
        self.time_slice_button.show()
        self.cross_line_button.show()
        self.inline_button.show()

        # Parse the key to determine the color mapping
        parts = key.split('_')
        if len(parts) == 1:
            self.color_mapp = 'seismic_default'
        else:
            self.color_mapp = '_'.join(parts[1:]).lower().replace(' ', '_')

        # Set the selected tensor as dataTensor
        self.tensor_data = self.tensor_dict[key]
        self.file_name = key
        self.file_path = self.loaded_file_paths[self.file_name.split('_')[0]]
        # Add the loaded file path to the dictionary
        print(f"Selected tensor: {key}")
        self.three_d()
        self.index_entry.setText("0")
        self.channel_index_entry.setText("0")

        if self.tensor_data.shape[-1] > 1:
            self.set_slider_range(self.tensor_data.shape[-1])
            self.channel_index_label.show()
            self.channel_index_entry.show()
            self.Channel_slider.show()
        else:
            self.channel_index_label.hide()
            self.channel_index_entry.hide()
            self.Channel_slider.hide()

        if len(self.tensor_data.shape) > 4:
            self.channel_label.show()
            self.channel_entry.show()
        else:
            self.channel_label.hide()
            self.channel_entry.hide()

        sampling_interval = self.get_sampling_interval_from_file_name()
        current_index = self.tab_widget.currentIndex()
        current_widget = self.tab_widget.widget(current_index)
        if current_widget.layout():
            canvas = current_widget.layout().itemAt(0).widget()

            if isinstance(canvas, PyQtGraphCanvas):
                # If the PyQtGraphCanvas is shown, hide the sampling interval UI elements
                self.sampling_interval_label.hide()
                self.sampling_interval_entry.hide()
                self.Three_D_button.show()
                self.plot_button.clicked.disconnect()  # Disconnect previous signal
                self.plot_button.clicked.connect(self.plot_data3d)
            elif isinstance(canvas, RoundedCanvas):
                # If the RoundedCanvas is shown
                self.plot_button.clicked.disconnect()  # Disconnect previous signal
                self.plot_button.clicked.connect(self.plot_tensor)
            elif isinstance(canvas, RoundedCanvas) and sampling_interval is None:
                # If the RoundedCanvas is shown and there is no sampling interval, show the UI elements
                self.sampling_interval_label.show()
                self.sampling_interval_entry.show()

        if sampling_interval is not None:
            self.sampling_interval_label.hide()
            self.sampling_interval_entry.hide()

        if self.file_path.endswith('.las'):
            self.sampling_interval_label.hide()
            self.sampling_interval_entry.hide()
            self.time_slice_button.hide()
            self.cross_line_button.hide()
            self.inline_button.hide()
            self.channel_index_entry.hide()
            self.channel_index_label.hide()
            self.Channel_slider.hide()
            self.Three_D_button.hide()
            self.plot_button.clicked.disconnect()  # Disconnect previous signal
            self.plot_button.clicked.connect(self.well_log_viewer)

    def add_tensor(self, key, tensor):
        # Add the tensor to the dictionary with a unique key
        self.tensor_dict[key] = tensor
        self.populateList()

        # Find the QTreeWidgetItem associated with the key
        parts = key.strip().split('_', 1)
        data_name = parts[0]
        attribute_name = parts[1] if len(parts) > 1 else ""

        # Find the parent item for the data
        parent_items = self.treeWidget.findItems(data_name, Qt.MatchExactly)
        parent_item = parent_items[0] if parent_items else None

        if parent_item:
            if attribute_name:
                # We are adding an attribute, so select the appropriate child item
                for i in range(parent_item.childCount()):
                    child_item = parent_item.child(i)
                    if child_item.text(0).strip() == attribute_name:
                        # Select the child item
                        self.treeWidget.setCurrentItem(child_item)
                        self.selectTensor(key)  # Manually invoke to ensure the tensor is set
                        break
            else:
                # We are adding a parent data, select the parent item
                self.treeWidget.setCurrentItem(parent_item)
                self.selectTensor(data_name)  # Manually invoke to ensure the tensor is set

    def load_las_file(self):
        # Open a file dialog to select an LAS file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(None, "Open LAS File", "", "LAS Files (*.las);;Text Files (*.txt)",
                                                   options=options)
        if file_name:
            # Extract the display name (name without extension)
            display_name = os.path.splitext(os.path.basename(file_name))[0]

            # Parse the LAS file
            csv_data = self.parse_las_file(file_name)

            # Store the data in the tensor dictionary with the display name as the key
            self.tensor_dict[display_name] = csv_data
            self.loaded_file_paths[display_name] = file_name

            # Use existing function to add the entry to the tree widget
            self.add_tensor(display_name, csv_data)
            self.update_recent_files_submenu()

    def parse_las_file(self, file_path):
        headers = []
        units = {}
        data_rows = []
        current_file_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, 'r') as file:
            lines = file.readlines()
            data_section = False

            for line in lines:
                line = line.strip()

                if line.startswith("~Curve"):
                    # Beginning of headers and units section
                    headers, units = self.extract_headers_and_units(lines)
                elif line.startswith("~Ascii"):
                    # Beginning of data section
                    data_section = True
                elif data_section:
                    # Collect data rows
                    data_rows.append(line.split())

        # Store units in the dictionary using the file name as the key
        self.units_dict[current_file_name] = units

        # Convert the collected data to a DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        return df

    def extract_headers_and_units(self, lines):
        headers = []
        units = {}
        curve_section = False

        for line in lines:
            line = line.strip()

            if line.startswith("~Curve"):
                curve_section = True
            elif line.startswith("~Parameter") or line.startswith("~Ascii"):
                break  # End of header section

            if curve_section and not line.startswith("#") and line and '.' in line:
                # Extracting headers and units
                parts = line.split()
                header_name = parts[0].replace(".", "")
                unit = parts[1] if len(parts) > 1 else ""
                headers.append(header_name)
                units[header_name] = unit

        return headers, units

    def show_csv_in_dialog(self):
        """
        Display the CSV data in a new non-modal dialog window.
        """
        # Create a dialog window
        dialog = QDialog(self)
        dialog.setWindowTitle("CSV Viewer")
        dialog.setWindowModality(Qt.NonModal)  # Ensure the dialog is non-modal
        dialog.resize(800, 600)  # Set initial size

        # Set window flags to enable maximize button and resizing
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)

        # Create a layout for the dialog
        layout = QVBoxLayout(dialog)

        # Create a QTableView to display the CSV data
        table_view = QTableView(dialog)
        table_view.setModel(self.create_pandas_model())
        table_view.setSelectionBehavior(QAbstractItemView.SelectRows)  # Optional: select entire rows

        # Add the table view to the layout
        layout.addWidget(table_view)

        # Set the layout for the dialog
        dialog.setLayout(layout)

        # Show the dialog
        dialog.show()  # Show the dialog maximized

    def create_pandas_model(self):
        """
        Convert the CSV data in self.tensor_data to a QAbstractTableModel for use with QTableView.

        :return: A QAbstractTableModel that can be used with QTableView.
        """

        class PandasModel(QAbstractTableModel):
            def __init__(self, data):
                super(PandasModel, self).__init__()
                self._data = data

            def rowCount(self, parent=None):
                return self._data.shape[0]

            def columnCount(self, parent=None):
                return self._data.shape[1]

            def data(self, index, role=Qt.DisplayRole):
                if role == Qt.DisplayRole:
                    return str(self._data.iloc[index.row(), index.column()])
                return None

            def headerData(self, section, orientation, role=Qt.DisplayRole):
                if role == Qt.DisplayRole:
                    if orientation == Qt.Horizontal:
                        return str(self._data.columns[section])
                    if orientation == Qt.Vertical:
                        return str(self._data.index[section])
                return None

        # Create and return a PandasModel instance
        return PandasModel(self.tensor_data)

    def well_log_viewer(self):
        """Open a new non-modal dialog window with well log plots."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Well Log Viewer")
        dialog.setGeometry(100, 100, 1200, 800)

        # Ensure the dialog can be maximized and minimized independently
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)

        # Create a scroll area for horizontal scrolling
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Create a central widget and layout for the scroll area
        central_widget = QWidget()
        h_layout = QHBoxLayout(central_widget)  # Use horizontal layout to align graphs side by side

        # Set the scroll area's central widget
        scroll_area.setWidget(central_widget)

        # Create the main layout for the dialog window
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)

        # Set the main layout for the dialog
        dialog.setLayout(main_layout)

        # Plot the well log data
        self.create_well_log_plots(h_layout)

        # Show the dialog in non-modal mode
        dialog.show()

    def identify_depth_column(self):
        """Identify the depth column, typically the first column, and return its name."""
        possible_names = ['DEPT', 'DEPTH', 'Depth', 'depth']
        for name in possible_names:
            if name in self.tensor_data.columns:
                return name
        return self.tensor_data.columns[0]  # If no common name matches, assume the first column is depth

    def create_well_log_plots(self, h_layout):
        # Identify the key for the active data (self.tensor_data) in the tensor_dict
        active_data_key = None

        for key, value in self.tensor_dict.items():
            if value is self.tensor_data:
                active_data_key = key
                break

        # Filter the items to be shown in the list widget
        csv_items = [key for key, value in self.tensor_dict.items() if
                     value is not self.tensor_data and isinstance(value, pd.DataFrame)]

        # If there are additional wells
        if csv_items:

            # Ask the user if they want to select additional wells
            confirm_selection = QMessageBox.question(
                self,
                "Select Additional Wells",
                "Do you want to select additional wells?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            # If the user selects 'yes'
            if confirm_selection == QMessageBox.Yes:

                # Create a new dialog to select additional CSV files
                csv_selection_dialog = QDialog(self)
                csv_selection_dialog.setWindowTitle("Select Additional Wells")

                layout = QVBoxLayout(csv_selection_dialog)

                csv_list_widget = QListWidget()
                csv_list_widget.setSelectionMode(QListWidget.MultiSelection)

                # Add items to the list, excluding the currently active self.tensor_data
                for key, value in self.tensor_dict.items():
                    if value is not self.tensor_data:  # Skip the currently active tensor data
                        if isinstance(value, pd.DataFrame):
                            csv_list_widget.addItem(key)
                layout.addWidget(csv_list_widget)

                select_button = QPushButton("OK")
                select_button.setFocusPolicy(Qt.NoFocus)
                select_button.clicked.connect(csv_selection_dialog.accept)  # Close the current dialog
                layout.addWidget(select_button)

                csv_selection_dialog.setLayout(layout)
                csv_selection_dialog.exec()  # Use exec() to block until the dialog is closed

                # Get selected additional CSV files
                selected_items = csv_list_widget.selectedItems()
                selected_files = [item.text() for item in selected_items]
            else:
                selected_files = []
        else:
            selected_files = []

        def valid_columns(df):
            """Return columns that are not entirely NaN or negative after converting to numeric."""
            valid_columns = []
            for col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # Check if the column is not entirely NaN and does not contain only negative values
                if not numeric_col.isna().all() and not (numeric_col < 0).all():
                    valid_columns.append(col)
            return valid_columns

        # Determine the data frames to be plotted
        if len(selected_files) == 0:
            # No additional files selected, only plot the active tensor data
            data_frames = [self.tensor_data]
            keys = [active_data_key]
        else:
            # Include the currently active tensor data and the selected additional files
            data_frames = [self.tensor_data] + [self.tensor_dict[file] for file in selected_files]
            keys = [active_data_key] + selected_files

        # Determine common columns across all selected files, including the active tensor data
        common_columns = set(valid_columns(data_frames[0]))
        for df in data_frames[1:]:
            common_columns.intersection_update(valid_columns(df))

        if not common_columns:
            print("No common columns found among the selected CSV files.")
            return

        # Remove the depth column from common columns
        depth_col = self.identify_depth_column()
        common_columns.discard(depth_col)
        columns = list(common_columns)

        # Create well log plots for each common column, aligned by dataset
        for col in columns:
            for data_frame, file in zip(data_frames, keys):
                depth_data = pd.to_numeric(data_frame[depth_col],
                                           errors='coerce')  # Convert to numeric, force NaNs on errors
                valid_depth_mask = (depth_data.notna()) & (depth_data != -999.25)  # Filter valid depth data
                depth_data = depth_data[valid_depth_mask]

                data_col = pd.to_numeric(data_frame[col],
                                         errors='coerce')  # Convert the data to numeric, forcing NaNs for non-numeric values
                data_col = data_col[valid_depth_mask]  # Apply the same valid mask to other columns
                data_col.replace(-999.25, np.nan, inplace=True)  # Treat -999.25 as NaN

                if data_col.isna().all():
                    # Skip plotting if the entire column is non-numeric or NaN
                    print(f"Skipping non-numeric or invalid column: {col} in file {file}")
                    continue

                # Create a separate plot for each column using a standard PlotWidget
                plot_canvas = pg.PlotWidget()
                plot_canvas.setMenuEnabled(False)
                plot_canvas.setBackground(self.last_selected_color)  # Set background color

                # Plot the data
                plot_canvas.plot(data_col, depth_data, pen=pg.mkPen(color=(75, 75, 75), width=2))

                # Set labels and invert Y axis for depth
                plot_canvas.setLabel('left', 'Depth (m)')
                plot_canvas.setLabel('top', f"{col} ({file})")  # Include the file name or key in the label for clarity
                plot_canvas.invertY(True)

                # Set x-axis range based on the data
                if not data_col.dropna().empty:
                    x_min = data_col.min()
                    x_max = data_col.max()
                    plot_canvas.setXRange(x_min, x_max)
                    units = self.units_dict.get(file, {})
                    unit_label = f"({units.get(col, '')})" if col in units else ""
                    plot_canvas.setLabel('top', f"{col} {unit_label} ({file})")

                # Remove the x-axis at the bottom
                plot_canvas.getAxis('bottom').setVisible(False)

                # Show grid lines
                plot_canvas.showGrid(x=True, y=True)

                # Adjust the plot width
                plot_canvas.setFixedWidth(300)  # Adjust the width as needed

                # Add each plot to the horizontal layout
                h_layout.addWidget(plot_canvas)

                # Link the y-axis (depth) to the first plot for synchronized scrolling
                if h_layout.count() > 1:
                    plot_canvas.setYLink(h_layout.itemAt(0).widget())

    def well_context_menu_event(self, event):
        # Create the main context menu
        menu = QMenu(self)

        # "Color Mapping" menu item with sub-menu
        colormap_menu = menu.addMenu("Color Mapping")

        # List of colormap options
        color_mappings = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'jet', 'hsv', 'hot', 'cool', 'spring',
            'summer', 'autumn', 'winter', 'bone', 'copper', 'tab20'
        ]

        # Add colormap options to the submenu
        for color_mapping in color_mappings:
            action = QAction(color_mapping, self.plot_widget)
            action.triggered.connect(lambda checked, cm=color_mapping: self.change_well_colormap(cm))
            colormap_menu.addAction(action)

        # "Export" menu item
        export_action = QAction("Export", self.plot_widget)
        export_action.triggered.connect(self.show_export_dialog)
        menu.addAction(export_action)

        # Add the colormap submenu to the main menu
        menu.addMenu(colormap_menu)

        # Show the context menu at the position of the event
        menu.exec_(event.screenPos())

    def show_export_dialog(self):
        # Create and show the export dialog
        self.export_well_dialog = exportDialog.ExportDialog(self.plot_widget.scene())
        self.export_well_dialog.show(self.plot_widget.plotItem)

    def change_well_colormap(self, color_mapping):
        self.well_color_mapping = color_mapping

        # Use a colormap with sufficient distinct colors
        cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')

        # Map labels to colors and convert to QColor for pyqtgraph
        colors = cmap.map(self.well_color_data, mode='qcolor')

        # Create a list of brushes for each color
        brushes = list(map(pg.mkBrush, colors))

        # Update the brushes of the existing scatter plot item
        scatter = self.plot_widget.listDataItems()[0]  # Assuming the scatter plot is the first item
        scatter.setBrush(brushes)

        # Update the ColorBarItem for the color bar
        if hasattr(self, 'well_color_bar') and self.well_color_bar is not None:
            self.well_color_bar.setColorMap(cmap)

    def open_plot_dialog(self):
        dialog = QDialog(self)
        dialog.resize(1000, 600)
        dialog.setWindowTitle("Well Analysis")

        # Ensure the dialog can be maximized and minimized independently
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)

        # Set up the main layout
        splitter = QSplitter(Qt.Horizontal, dialog)

        # Left layout: Scrollable results list, ComboBox, Plot button
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Scrollable area for results
        result_area = QScrollArea()
        result_area.setWidgetResizable(True)
        result_area.setFixedHeight(200)  # Set the height to a shorter value
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_area.setWidget(result_widget)
        left_layout.addWidget(result_area)

        def update_button_text():
            if analysis_type_combo.currentText() == "Predict":
                plot_button.setText("Run")
            else:
                plot_button.setText("Plot")

        # ComboBox for selecting analysis type
        analysis_type_combo = QComboBox()
        analysis_type_combo.addItems(["Cross Plots", "Regression", "Clustering", "Predict"])
        analysis_type_combo.currentIndexChanged.connect(update_button_text)
        left_layout.addWidget(analysis_type_combo)

        # Plot button
        plot_button = QPushButton("Plot")
        plot_button.setFocusPolicy(Qt.NoFocus)
        plot_button.clicked.connect(lambda: self.handle_plot_button(dialog, analysis_type_combo, result_layout))
        left_layout.addWidget(plot_button)

        # Spacer to push everything to the top
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer)

        splitter.addWidget(left_widget)  # Add the left layout to the splitter

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Right layout: PyQtGraph plot area
        self.plot_widget = pg.PlotWidget()  # Assuming self.plot_widget is still a class attribute
        self.plot_widget.setBackground(self.last_selected_color)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.scene().contextMenuEvent = self.well_context_menu_event
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        # Add padding around the plot widget
        right_layout.setContentsMargins(10, 10, 10, 10)  # Add margins (left, top, right, bottom) as needed

        right_layout.addWidget(self.plot_widget)
        splitter.addWidget(right_widget)  # Add the right layout to the splitter

        # Set initial sizes (ratios) for the splitter
        splitter.setSizes([300, 700])  # Adjust these values as needed to set the initial ratio

        # Use the splitter as the main layout
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.addWidget(splitter)
        dialog.setLayout(dialog_layout)

        dialog.finished.connect(self.reset_well_color_bar)

        dialog.show()

    def reset_well_color_bar(self):
        self.well_color_bar = None

    def handle_plot_button(self, parent_dialog, analysis_type_combo, result_layout):
        # Determine the selected analysis type
        analysis_type = analysis_type_combo.currentText()

        if analysis_type == "Cross Plots":
            self.open_cross_plot_options(parent_dialog, result_layout, analysis_type_combo)
        elif analysis_type == "Regression":
            self.open_regression_options(parent_dialog, result_layout, analysis_type_combo)
        elif analysis_type == "Clustering":
            self.open_clustering_options(parent_dialog, result_layout, analysis_type_combo)
        elif analysis_type == "Predict":
            self.process_and_predict()

    def process_and_predict(self):
        try:
            # Step 1: CSV File Selection
            csv_selection_dialog = QDialog(self)
            csv_selection_dialog.setWindowTitle("Select CSV File")
            layout = QVBoxLayout(csv_selection_dialog)

            csv_list_widget = QListWidget()
            csv_list_widget.setSelectionMode(QListWidget.SingleSelection)
            for key in self.tensor_dict:
                if isinstance(self.tensor_dict[key], pd.DataFrame):
                    csv_list_widget.addItem(key)
            layout.addWidget(csv_list_widget)

            select_button = QPushButton("OK")
            select_button.setFocusPolicy(Qt.NoFocus)
            select_button.clicked.connect(csv_selection_dialog.accept)
            layout.addWidget(select_button)

            csv_selection_dialog.setLayout(layout)
            csv_selection_dialog.exec()

            selected_items = csv_list_widget.selectedItems()
            if len(selected_items) == 0:
                print("Please select one CSV file.")
                return

            selected_file = selected_items[0].text()
            selected_df = self.tensor_dict[selected_file]

            def valid_columns(df):
                """Return columns that are not entirely NaN or negative after converting to numeric."""
                valid_columns = []
                for col in df.columns:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    # Check if the column is not entirely NaN and does not contain only negative values
                    if not numeric_col.isna().all() and not (numeric_col < 0).all():
                        valid_columns.append(col)
                return valid_columns

            # If only one file is selected, directly use its columns
            data_frame = self.tensor_dict[selected_file[0]]
            columns = valid_columns(data_frame)

            feature_input_dialog = QDialog(self)
            feature_input_dialog.setWindowTitle("Select Feature Inputs")
            layout = QVBoxLayout(feature_input_dialog)

            feature_input_list = QListWidget()
            feature_input_list.setSelectionMode(QListWidget.MultiSelection)
            for col in columns:
                feature_input_list.addItem(col)
            layout.addWidget(feature_input_list)

            select_button = QPushButton("OK")
            select_button.setFocusPolicy(Qt.NoFocus)
            select_button.clicked.connect(feature_input_dialog.accept)
            layout.addWidget(select_button)

            feature_input_dialog.setLayout(layout)
            feature_input_dialog.exec()

            selected_feature_items = feature_input_list.selectedItems()
            if len(selected_feature_items) == 0:
                print("Please select at least one feature input.")
                return

            selected_features = [item.text() for item in selected_feature_items]

            # Step 3: Find Target Variables Based on Selected Features in the Model Dictionary
            potential_target_variables = set()
            for model_type, models in self.model_dict.items():
                for model_info in models:
                    if set(selected_features) == set(model_info['features']):
                        potential_target_variables.add(model_info['target'])

            if not potential_target_variables:
                QMessageBox.warning(self, "No Matching Models",
                                    "No models are available for the selected feature inputs.")
                return

            # Step 4: Target Variable Selection from Model Dictionary
            target_variable_dialog = QDialog(self)
            target_variable_dialog.setWindowTitle("Select Target Variable")
            layout = QVBoxLayout(target_variable_dialog)

            target_variable_list = QListWidget()
            target_variable_list.setSelectionMode(QListWidget.SingleSelection)
            for target in potential_target_variables:
                target_variable_list.addItem(target)
            layout.addWidget(target_variable_list)

            select_button = QPushButton("OK")
            select_button.setFocusPolicy(Qt.NoFocus)
            select_button.clicked.connect(target_variable_dialog.accept)
            layout.addWidget(select_button)

            target_variable_dialog.setLayout(layout)
            target_variable_dialog.exec()

            selected_target_items = target_variable_list.selectedItems()
            if len(selected_target_items) == 0:
                print("Please select a target variable.")
                return

            selected_target = selected_target_items[0].text()

            # Step 5: Model Selection Based on Features and Target Variable
            available_models = []
            for model_type, models in self.model_dict.items():
                for model_info in models:
                    if set(selected_features) == set(model_info['features']) and selected_target == model_info['target']:
                        available_models.append((model_type, model_info))

            if not available_models:
                QMessageBox.warning(self, "No Matching Model",
                                    "No trained models available for the selected features and target variable.")
                return

            model_selection_dialog = QDialog(self)
            model_selection_dialog.setWindowTitle("Select Model for Prediction")
            layout = QVBoxLayout(model_selection_dialog)

            model_list_widget = QListWidget()
            model_list_widget.setSelectionMode(QListWidget.SingleSelection)
            for model_type, model_info in available_models:
                # Use a string representation for easy selection, could be more detailed if needed
                model_list_widget.addItem(f"{model_type} model")

            layout.addWidget(model_list_widget)

            select_button = QPushButton("OK")
            select_button.setFocusPolicy(Qt.NoFocus)
            select_button.clicked.connect(model_selection_dialog.accept)
            layout.addWidget(select_button)

            model_selection_dialog.setLayout(layout)
            model_selection_dialog.exec()

            selected_model_items = model_list_widget.selectedItems()
            if len(selected_model_items) == 0:
                print("Please select a model.")
                return

            # Extract the selected model information
            selected_model_index = model_list_widget.currentRow()
            selected_model_type, selected_model_info = available_models[selected_model_index]

            # Get the model object
            model = selected_model_info['model']
            file_path = selected_model_info['file_path']
            dimensionality_reduction_type_combo = selected_model_info['dim_type']
            n_components = selected_model_info['n_components']

            # Step 6: Data Preprocessing
            # Check for corresponding samples in selected columns
            if selected_features:
                # Read selected CSV
                combined_df = selected_df[selected_features]

                # Convert selected columns to numeric and exclude NaNs and -999.25
                combined_df[selected_features].apply(pd.to_numeric, errors='coerce')
                combined_df = combined_df.replace(-999.25, np.nan)

                # Count rows where all selected columns have valid (non-NaN) values
                valid_rows = combined_df.dropna(subset=selected_features)
                num_valid_points = len(valid_rows)

                if num_valid_points == 0:
                    QMessageBox.warning(self, "No Valid Data Points",
                                        "No data points have values for all selected columns.")
                    return
                else:
                    print(f"Number of data points with values for all selected columns: {num_valid_points}")

            dialog = QDialog(self)
            dialog.setWindowTitle("Select Normalization Type")
            layout = QVBoxLayout()

            label = QLabel("Choose normalization type:")
            layout.addWidget(label)

            combo = QComboBox()
            combo.addItems(["Min-Max Scaler", "Standard Scaler", "None"])
            layout.addWidget(combo)

            button = QPushButton("OK")
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(dialog.accept)
            layout.addWidget(button)

            dialog.setLayout(layout)
            dialog.setMinimumWidth(225)
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.adjustSize()
            dialog.exec()

            normalization_choice = combo.currentText()

            X = valid_rows[selected_features]

            if normalization_choice == "Min-Max Scaler":
                scaler = MinMaxScaler()
                X_clean = scaler.fit_transform(X)
            elif normalization_choice == "Standard Scaler":
                scaler = StandardScaler()
                X_clean = scaler.fit_transform(X)
            else:
                X_clean = X.to_numpy()

            QApplication.setOverrideCursor(self.custom_cursor)

            if dimensionality_reduction_type_combo == "PCA":
                # Create PCA object to retain 90% of variance
                pca = PCA(n_components=n_components)

                # Fit PCA model to data
                pca.fit(X_clean)

                X_clean = pca.transform(X_clean)

            elif dimensionality_reduction_type_combo == "t-SNE":
                # Create t-SNE object
                tsne = TSNE(n_components=n_components)
                # Fit and transform data
                X_clean = tsne.fit_transform(X_clean)

            elif dimensionality_reduction_type_combo == "ICA":
                # Create ICA object
                ica = FastICA(n_components=n_components, max_iter=500)
                # Fit ICA model to data
                ica.fit(X_clean)
                X_clean = ica.transform(X_clean)

            elif dimensionality_reduction_type_combo == "Isomap":
                # Create Isomap object
                isomap = Isomap(n_components=n_components)
                # Fit and transform data
                X_clean = isomap.fit_transform(X_clean)

            elif dimensionality_reduction_type_combo == "Random Projection":
                # Create Random Projection object
                grp = GaussianRandomProjection(n_components=n_components)
                # Fit and transform data
                X_clean = grp.fit_transform(X_clean)
            elif dimensionality_reduction_type_combo == "Autoencoder":
                # Define the input dimension based on X_queen's shape
                input_dim = X_clean.shape[1]

                # Split the data into training and validation sets
                X_train, X_val = train_test_split(X_clean, test_size=0.2, random_state=42)

                # Define the encoding dimension
                encoding_dim = max(1, n_components)

                # Input layer
                input_layer = Input(shape=(input_dim,))

                # Encoder: Dense layers
                x = Dense(128, activation='relu')(input_layer)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)

                # Check if we can reshape it properly for Conv1D
                if x.shape[1] == input_dim:  # Only reshape if it matches input_dim
                    x = Reshape((input_dim, 1))(x)  # Reshape for Conv1D
                    x = Conv1D(32, 3, activation='relu', padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Dropout(0.2)(x)
                    x = Flatten()(x)  # Flatten to pass to dense layers
                else:
                    # Skip Conv1D if reshape isn't possible directly from Dense
                    x = Dense(64, activation='relu')(x)

                # Multi-Head Attention for capturing different "views" or dependencies
                x = Reshape((1, -1))(x)  # Reshape to (batch_size, 1, features)
                attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                attention_output = Dropout(0.2)(attention_output)
                attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)  # Residual connection
                attention_output = Reshape((-1,))(attention_output)  # Flatten

                # Final dense layer to reduce to the encoding dimension
                encoded = Dense(encoding_dim, activation='relu')(attention_output)

                # Decoder: Reverse the process
                x = Dense(128, activation='relu')(encoded)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)

                # Adding Dense layers
                x = Dense(64, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)

                # Reconstructing the original input dimensions
                decoded = Dense(input_dim, activation='sigmoid', dtype='float32')(x)

                # Autoencoder model
                autoencoder = Model(input_layer, decoded)

                # Compile the model
                autoencoder.compile(optimizer='adam', loss='mean_squared_error')

                # Encoder model to reduce dimensionality
                encoder = Model(input_layer, encoded)

                # Summary of the autoencoder
                autoencoder.summary()

                # Train the autoencoder with training-validation split
                autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True,
                                validation_data=(X_val, X_val))

                # Use the encoder to transform the data
                X_clean = encoder.predict(X_clean)

            else:
                pass

            selected_model_type = selected_model_type.split('_')[0]

            if selected_model_type in ["ResNet", "TCN", "Keras Regressor", "TableNet"]:
                completion_event = multiprocessing.Event()
                # Create queues for communication
                input_queue = multiprocessing.Queue()
                output_queue = multiprocessing.Queue()
                # Prepare the input data (e.g., input_data)
                input_data = X_clean  # The data you need to pass to the process
                # Put the input data into the input queue
                input_queue.put(input_data)

                process = multiprocessing.Process(target=TensorVisualizer.run_well_inference_neural_net, args=(
                    input_queue, output_queue, selected_model_type, completion_event, file_path))

                # Start the process
                process.start()

                # Wait for the process to signal completion
                while not completion_event.is_set():
                    process.join(timeout=1)  # Check every second if the event is set

                # Wait for the process to finish
                y_pred = output_queue.get()

                # Terminate the process if it's still alive
                if process.is_alive():
                    print("Process finished, terminating...")
                    process.terminate()
                    process.join()  # Ensure the process has been terminated
            else:
                y_pred = model.predict(X_clean)

            # Step 8: Update CSV with Predictions
            predicted_col_name = f"{selected_target}_predicted"
            selected_df[predicted_col_name] = pd.NA
            selected_df[predicted_col_name] = pd.Series(y_pred, index=valid_rows.index)

            self.tensor_data = selected_df
            # Store the modified tensor with an informative key
            self.add_tensor(f"{selected_file}_Predicted", self.tensor_data)
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Prediction Complete", f"Predictions added to {selected_file}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    @staticmethod
    def run_well_inference_neural_net(input_queue, output_queue, selected_model_type, completion_event, file_path):
        try:
            X_clean = input_queue.get()

            def load_model_by_type(file_path, selected_model_type):
                """
                Loads the model from the specified file path based on its type.

                :param file_path: Path to the saved model file.
                :param selected_model_type: Type of the model (e.g., "Keras", "PyTorch", "Sklearn").
                :return: The loaded model object.
                """
                if selected_model_type in ["Keras Regressor"]:
                    # Load the Keras model
                    return load_model(file_path, compile=False)
                elif selected_model_type in ["ResNet", "TCN", "TableNet"]:
                    # Load the whole PyTorch model
                    return torch.load(file_path)

            # Predict using the fitted model
            if selected_model_type == "ResNet":
                model = load_model_by_type(file_path, selected_model_type)
                # Make predictions
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_val_tensor = torch.tensor(X_clean.reshape(X_clean.shape[0], 1, -1), dtype=torch.float32)
                with torch.no_grad():
                    y_pred = model(
                        X_val_tensor.to(device)).cpu().numpy()  # Move predictions back to CPU for compatibility
                    y_pred = np.array(y_pred).squeeze()
            elif selected_model_type == "TableNet":
                model = load_model_by_type(file_path, selected_model_type)
                y_pred = model.predict(X_clean)
                y_pred = np.array(y_pred).squeeze()
            elif selected_model_type == "TCN":
                model = load_model_by_type(file_path, selected_model_type)
                # Make predictions
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_val_tensor = torch.tensor(X_clean.reshape(X_clean.shape[0], 1, -1), dtype=torch.float32)
                with torch.no_grad():
                    y_pred = model(
                        X_val_tensor.to(device)).cpu().numpy()  # Move predictions back to CPU for compatibility
                    y_pred = np.array(y_pred).squeeze()
            else:
                model = load_model_by_type(file_path, selected_model_type)
                y_pred = model.predict(X_clean)
                y_pred = np.array(y_pred).squeeze()

            # Send the result back via the output queue
            output_queue.put(y_pred)

            # Signal completion to the parent process
            completion_event.set()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical("Error", f"An error occurred: {e}")

    def open_clustering_options(self, parent_dialog, result_layout, analysis_type_combo):
        # Create a new dialog to select linear or non-linear regression
        cross_plot_dialog = QDialog(parent_dialog)
        cross_plot_dialog.setWindowTitle("Clustering Options")

        layout = QVBoxLayout(cross_plot_dialog)

        clustering_type_combo = QComboBox()
        clustering_type_combo.addItems(["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture", "Spectral",
                                        "Agglomerative", "Birch", "Affinity Propagation", "Mean Shift", "OPTICS",
                                        "SOM", "HDBSCAN", "DEC"])
        layout.addWidget(clustering_type_combo)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(cross_plot_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        cross_plot_dialog.setLayout(layout)
        cross_plot_dialog.adjustSize()
        cross_plot_dialog.setMinimumWidth(225)
        # Disable the '?' help button on the dialog
        cross_plot_dialog.setWindowFlags(cross_plot_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        result = cross_plot_dialog.exec()  # Use exec() to block until the dialog is closed
        if result == QDialog.Accepted:
            clustering_type_combo = clustering_type_combo.currentText()
            # Trigger the next function after the dialog is closed
            self.open_csv_selection_dialog(parent_dialog, clustering_type_combo, result_layout, analysis_type_combo)

    def open_regression_options(self, parent_dialog, result_layout, analysis_type_combo):

        # Create a new dialog to select linear or non-linear regression
        cross_plot_dialog = QDialog(parent_dialog)
        cross_plot_dialog.setWindowTitle("Regression Options")

        layout = QVBoxLayout(cross_plot_dialog)

        regression_type_combo = QComboBox()
        regression_type_combo.addItems([
            "Ridge Regression",
            "Lasso Regression",
            "Bayesian Ridge Regression",
            "Random Forest Regressor",
            "Support Vector Regressor",
            "K-Nearest Neighbors Regressor",
            "Decision Tree Regressor",
            "Gaussian Process Regressor",
            "XGBoost Regressor",
            "LightGBM Regressor",
            "NGBoost Regressor",
            "CatBoost Regressor",
            "Gradient Boosting Regressor",
            "AdaBoost Regressor",
            "Keras Regressor",
            "ResNet",
            "TableNet",
            "TCN",
        ])
        layout.addWidget(regression_type_combo)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(cross_plot_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        cross_plot_dialog.setLayout(layout)
        cross_plot_dialog.adjustSize()
        cross_plot_dialog.setMinimumWidth(225)
        # Disable the '?' help button on the dialog
        cross_plot_dialog.setWindowFlags(cross_plot_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        result = cross_plot_dialog.exec()  # Use exec() to block until the dialog is closed
        if result == QDialog.Accepted:
            regression_type_combo = regression_type_combo.currentText()
            # Trigger the next function after the dialog is closed
            self.open_csv_selection_dialog(parent_dialog, regression_type_combo, result_layout, analysis_type_combo)

    def open_cross_plot_options(self, parent_dialog, result_layout, analysis_type_combo):
        # Create a new dialog to select linear or non-linear regression
        cross_plot_dialog = QDialog(parent_dialog)
        cross_plot_dialog.setWindowTitle("Cross Plot Options")

        layout = QVBoxLayout(cross_plot_dialog)

        regression_type_combo = QComboBox()
        regression_type_combo.addItems(["Linear Regression", "Non-Linear Regression"])
        layout.addWidget(regression_type_combo)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(cross_plot_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        cross_plot_dialog.setLayout(layout)
        cross_plot_dialog.adjustSize()
        cross_plot_dialog.setMinimumWidth(225)
        # Disable the '?' help button on the dialog
        cross_plot_dialog.setWindowFlags(cross_plot_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        result = cross_plot_dialog.exec()  # Use exec() to block until the dialog is closed
        if result == QDialog.Accepted:
            # Trigger the next function after the dialog is closed
            self.open_csv_selection_dialog(parent_dialog, regression_type_combo, result_layout, analysis_type_combo)

    def open_csv_selection_dialog(self, parent_dialog, regression_type_combo, result_layout, analysis_type_combo):
        # Create a new dialog to select CSV files
        csv_selection_dialog = QDialog(parent_dialog)
        csv_selection_dialog.setWindowTitle("Select Wells")

        layout = QVBoxLayout(csv_selection_dialog)

        csv_list_widget = QListWidget()
        csv_list_widget.setSelectionMode(QListWidget.MultiSelection)
        for key in self.tensor_dict:
            if isinstance(self.tensor_dict[key], pd.DataFrame):
                csv_list_widget.addItem(key)
        layout.addWidget(csv_list_widget)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(csv_selection_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        csv_selection_dialog.setLayout(layout)
        csv_selection_dialog.exec()  # Use exec() to block until the dialog is closed

        # Trigger the next function after the dialog is closed
        self.show_common_columns(parent_dialog, csv_list_widget, regression_type_combo, result_layout, analysis_type_combo)

    def show_common_columns(self, parent_dialog, csv_list_widget, regression_type_combo, result_layout, analysis_type_combo):
        # Get selected CSV files and find common columns
        selected_items = csv_list_widget.selectedItems()
        selected_files = [item.text() for item in selected_items]

        if len(selected_files) == 0:
            print("Please select at least one CSV file.")
            return

        def valid_columns(df):
            """Return columns that are not entirely NaN or negative after converting to numeric."""
            valid_columns = []
            for col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # Check if the column is not entirely NaN and does not contain only negative values
                if not numeric_col.isna().all() and not (numeric_col < 0).all():
                    valid_columns.append(col)
            return valid_columns

        if len(selected_files) == 1:
            # If only one file is selected, directly use its columns
            data_frame = self.tensor_dict[selected_files[0]]
            columns = valid_columns(data_frame)
        else:
            # If multiple files are selected, find common columns
            data_frames = [self.tensor_dict[file] for file in selected_files]
            common_columns = set(valid_columns(data_frames[0]))
            for df in data_frames[1:]:
                common_columns.intersection_update(valid_columns(df))
            if not common_columns:
                print("No common columns found among the selected CSV files.")
                return
            columns = list(common_columns)

        common_columns_dialog = QDialog(parent_dialog)
        common_columns_dialog.setWindowTitle("Select Columns")

        layout = QVBoxLayout(common_columns_dialog)

        common_columns_list = QListWidget()
        common_columns_list.setSelectionMode(QListWidget.MultiSelection)
        for col in columns:
            common_columns_list.addItem(col)
        layout.addWidget(common_columns_list)

        plot_button = QPushButton("OK")
        plot_button.setFocusPolicy(Qt.NoFocus)
        plot_button.clicked.connect(common_columns_dialog.accept)  # Close the current dialog
        layout.addWidget(plot_button)

        common_columns_dialog.setLayout(layout)
        common_columns_dialog.exec()  # Use exec() to block until the dialog is closed
        # Get selected feature inputs
        selected_features = [item.text() for item in common_columns_list.selectedItems()]

        # Check for corresponding samples in selected columns
        if selected_features:
            # Read all selected CSVs into a combined DataFrame
            combined_df = pd.concat([self.tensor_dict[file] for file in selected_files])

            # Convert selected columns to numeric and exclude NaNs and -999.25
            combined_df[selected_features] = combined_df[selected_features].apply(pd.to_numeric, errors='coerce')
            combined_df = combined_df.replace(-999.25, np.nan)

            # Count rows where all selected columns have valid (non-NaN) values
            valid_rows = combined_df.dropna(subset=selected_features)
            num_valid_points = len(valid_rows)

            if num_valid_points == 0:
                QMessageBox.warning(parent_dialog, "No Valid Data Points",
                                    "No data points have values for all selected columns.")
                return
            else:
                print(f"Number of data points with values for all selected columns: {num_valid_points}")

            analysis_type = analysis_type_combo.currentText()

            if analysis_type == "Cross Plots":
                self.plot_cross_plot(selected_files, common_columns_list, regression_type_combo, result_layout,
                                     parent_dialog, num_valid_points)
            if analysis_type == "Clustering":

                self.run_Clustering(parent_dialog, selected_files, selected_features, regression_type_combo,
                                    result_layout, num_valid_points)
            elif analysis_type == "Regression":

                remaining_columns = list(set(columns) - set(selected_features))

                if not selected_features:
                    print("Please select at least one feature input.")
                    return
                if not remaining_columns:
                    print("No remaining columns available for the target variable.")
                    return

                # Proceed to select the target variable from the remaining columns
                self.select_target_variable(parent_dialog, selected_files, selected_features, remaining_columns,
                                            regression_type_combo, result_layout, analysis_type)

        else:
            print("No columns selected.")

    def run_Clustering(self, parent_dialog, selected_files, selected_features, regression_type_combo, result_layout, num_valid_points):

        try:
            dialog = QDialog(parent_dialog)
            dialog.setWindowTitle("Select Normalization Type")
            layout = QVBoxLayout()

            label = QLabel("Choose normalization type:")
            layout.addWidget(label)

            combo = QComboBox()
            combo.addItems(["Min-Max Scaler", "Standard Scaler", "None"])
            layout.addWidget(combo)

            button = QPushButton("OK")
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(dialog.accept)
            layout.addWidget(button)

            dialog.setLayout(layout)
            dialog.setMinimumWidth(225)
            # Disable the '?' help button on the dialog
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.adjustSize()
            if dialog.exec_() == QDialog.Accepted:
                normalization_choice = combo.currentText()
            else:
                normalization_choice = "None"

            # Load and merge data from the selected CSV files
            data_frames = [self.tensor_dict[file] for file in selected_files]
            merged_data = pd.concat(data_frames, join='inner', ignore_index=True)

            # Replace -999.25 with NaN
            merged_data.replace(-999.25, pd.NA, inplace=True)

            # Extract feature inputs and target variable
            X = merged_data[selected_features]

            # Convert all columns to numeric, forcing errors to NaN
            X = X.apply(pd.to_numeric, errors='coerce')

            # Replace negative values with NaN
            X = X.applymap(lambda x: pd.NA if x < 0 else x)

            # Drop rows with NaN values
            X = X.dropna()

            # Extract the cleaned feature inputs and target variable
            X_clean = X[selected_features]
            X_clean_copy = X_clean.copy()

            # Normalize the feature inputs based on the user's choice
            if normalization_choice == "Min-Max Scaler":
                scaler = MinMaxScaler()
                X_clean = scaler.fit_transform(X_clean)
            elif normalization_choice == "Standard Scaler":
                scaler = StandardScaler()
                X_clean = scaler.fit_transform(X_clean)

            # Convert to numpy arrays
            X_clean = np.asarray(X_clean, dtype=float)

            # User input simulation (for demonstration purposes)
            clustering_algorithm = regression_type_combo

            if clustering_algorithm == "DBSCAN":

                dialog = QDialog(parent_dialog)
                dialog.setWindowTitle("Parameters")
                layout = QVBoxLayout()

                label = QLabel("Choose the number of minimum samples")
                layout.addWidget(label)

                # Add a spin box for integer input
                spinBox = QSpinBox()
                spinBox.setMinimum(1)
                spinBox.setMaximum(99)
                spinBox.setValue(10)
                layout.addWidget(spinBox)

                epsilonlabel = QLabel("Choose Epsilon")
                layout.addWidget(epsilonlabel)

                # Add a spin box for integer input
                epsilonBox = QSpinBox()
                epsilonBox.setMinimum(1)
                epsilonBox.setMaximum(99)
                epsilonBox.setValue(10)
                layout.addWidget(epsilonBox)

                button = QPushButton("OK")
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                dialog.setMinimumWidth(225)
                # Disable the '?' help button on the dialog
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.adjustSize()
            elif clustering_algorithm == "OPTICS":
                dialog = QDialog(parent_dialog)
                dialog.setWindowTitle("Number of minimum samples")
                layout = QVBoxLayout()

                label = QLabel("Choose the number of clusters")
                layout.addWidget(label)

                # Add a spin box for integer input
                spinBox = QSpinBox()
                spinBox.setMinimum(1)
                spinBox.setMaximum(99)
                spinBox.setValue(10)
                layout.addWidget(spinBox)

                button = QPushButton("OK")
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                dialog.setMinimumWidth(225)
                # Disable the '?' help button on the dialog
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.adjustSize()
            elif clustering_algorithm == "SOM":
                dialog = QDialog(parent_dialog)
                dialog.setWindowTitle("Parameters")
                layout = QVBoxLayout()

                label = QLabel("Choose the X number")
                layout.addWidget(label)

                # Add a spin box for integer input
                xspinBox = QSpinBox()
                xspinBox.setMinimum(1)
                xspinBox.setMaximum(99)
                xspinBox.setValue(10)
                layout.addWidget(xspinBox)

                epsilonlabel = QLabel("Choose the Y number")
                layout.addWidget(epsilonlabel)

                # Add a spin box for integer input
                yBox = QSpinBox()
                yBox.setMinimum(1)
                yBox.setMaximum(99)
                yBox.setValue(10)
                layout.addWidget(yBox)

                label = QLabel("Choose Sigma")
                layout.addWidget(label)

                # Add a spin box for integer input
                SigmaspinBox = QSpinBox()
                SigmaspinBox.setMinimum(1)
                SigmaspinBox.setMaximum(99)
                SigmaspinBox.setValue(1)
                layout.addWidget(SigmaspinBox)

                epsilonlabel = QLabel("Choose learning_rate")
                layout.addWidget(epsilonlabel)

                # Add a spin box for integer input
                learning_rateBox = QSpinBox()
                learning_rateBox.setMinimum(1)
                learning_rateBox.setMaximum(99)
                learning_rateBox.setValue(50)
                layout.addWidget(learning_rateBox)

                button = QPushButton("OK")
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                dialog.setMinimumWidth(225)
                # Disable the '?' help button on the dialog
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.adjustSize()
            else:
                dialog = QDialog(parent_dialog)
                dialog.setWindowTitle("Number of clusters")
                layout = QVBoxLayout()

                label = QLabel("Choose the number of clusters")
                layout.addWidget(label)

                # Add a spin box for integer input
                spinBox = QSpinBox()
                spinBox.setMinimum(1)
                spinBox.setMaximum(99)
                spinBox.setValue(10)
                layout.addWidget(spinBox)

                button = QPushButton("OK")
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                dialog.setMinimumWidth(225)
                # Disable the '?' help button on the dialog
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.adjustSize()

            inertia = None

            if dialog.exec_() == QDialog.Accepted:
                if clustering_algorithm == "DBSCAN":
                    eps = epsilonBox.value()
                    n_clusters = spinBox.value()
                elif clustering_algorithm == "SOM":
                    x = xspinBox.value()
                    y = yBox.value()
                    learning_rate = learning_rateBox.value()
                    sigma = SigmaspinBox.value()
                else:
                    n_clusters = spinBox.value()

                if clustering_algorithm == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = model.fit_predict(X_clean)
                    inertia = model.inertia_

                elif clustering_algorithm == "Hierarchical":
                    Z = linkage(X_clean, method='ward')
                    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

                elif clustering_algorithm == "DBSCAN":
                    model = DBSCAN(eps=eps/100, min_samples=n_clusters)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Gaussian Mixture":
                    model = GaussianMixture(n_components=n_clusters, random_state=42)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Spectral":
                    model = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Agglomerative":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Birch":
                    model = Birch(n_clusters=n_clusters)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Affinity Propagation":
                    model = AffinityPropagation(random_state=42)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "Mean Shift":
                    model = MeanShift()
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "OPTICS":
                    model = OPTICS(min_samples=n_clusters)
                    labels = model.fit_predict(X_clean)
                elif clustering_algorithm == "HDBSCAN":
                    model = hdbscan.HDBSCAN(min_cluster_size=n_clusters)
                    labels = model.fit_predict(X_clean)

                elif clustering_algorithm == "SOM":
                    som = MiniSom(x=x, y=y, input_len=X_clean.shape[1], sigma=sigma, learning_rate=learning_rate / 100)
                    som.random_weights_init(X_clean)
                    som.train_random(X_clean, 100)
                    labels = np.array([som.winner(x)[1] for x in X_clean])

                elif clustering_algorithm == "DEC":
                    input_dim = X_clean.shape[1]
                    input_layer = Input(shape=(input_dim,))
                    encoder = Dense(500, activation='relu')(input_layer)
                    encoder = Dense(500, activation='relu')(encoder)
                    encoder = Dense(2000, activation='relu')(encoder)
                    encoder = Dense(10, activation='relu')(encoder)
                    decoder = Dense(2000, activation='relu')(encoder)
                    decoder = Dense(500, activation='relu')(decoder)
                    decoder = Dense(500, activation='relu')(decoder)
                    decoder = Dense(input_dim, activation='sigmoid')(decoder)

                    autoencoder = Model(inputs=input_layer, outputs=decoder)
                    autoencoder.compile(optimizer='adam', loss='mse')
                    autoencoder.fit(X_clean, X_clean, epochs=50, batch_size=256, shuffle=True)

                    encoder_model = Model(inputs=input_layer, outputs=encoder)
                    encoded_data = encoder_model.predict(X_clean)
                    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                    labels = kmeans.fit_predict(encoded_data)
                    inertia = kmeans.inertia_

                else:
                    raise ValueError("Unknown clustering algorithm: {}".format(clustering_algorithm))

                # Pass data to the plotting function
                self.plot_cluster_results(labels, X_clean_copy, clustering_algorithm, result_layout, selected_files,
                                          selected_features, parent_dialog, inertia, num_valid_points)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def plot_cluster_results(self, labels, X_clean_copy, clustering_algorithm, result_layout, selected_files,
                             selected_features, parent_dialog, inertia, num_valid_points):

        # Create a dialog to select the target variable
        target_variable_dialog = QDialog(parent_dialog)
        target_variable_dialog.setWindowTitle("Select plotting features")

        layout = QVBoxLayout(target_variable_dialog)

        label = QLabel()
        label.setText("Select two features:")
        layout.addWidget(label)

        target_variable_list = QListWidget()
        target_variable_list.setSelectionMode(QListWidget.MultiSelection)
        for col in selected_features:
            target_variable_list.addItem(col)
        layout.addWidget(target_variable_list)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(target_variable_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        target_variable_dialog.setLayout(layout)
        target_variable_dialog.exec()  # Use exec() to block until the dialog is closed

        # Get the selected target variable
        selected_target_items = target_variable_list.selectedItems()
        if len(selected_target_items) == 0:
            print("Please select a target variable.")
            return

        selected_item_x = selected_target_items[0].text()
        selected_item_y = selected_target_items[1].text()

        # Extract the cleaned feature inputs and target variable
        X_clean = X_clean_copy[selected_item_x]
        Y_clean = X_clean_copy[selected_item_y]

        # Convert to numpy arrays
        X_clean = np.asarray(X_clean, dtype=float)
        Y_clean = np.asarray(Y_clean, dtype=float)

        # Define a discrete colormap
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)

        self.well_color_data = labels / num_labels

        # Use a colormap with sufficient distinct colors
        cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')

        # Map labels to colors and convert to QColor for pyqtgraph
        colors = cmap.map(self.well_color_data, mode='qcolor')

        # Create a list of brushes for each color
        brushes = [pg.mkBrush(color) for color in colors]

        # Clear the previous plot and reset the view
        self.plot_widget.clear()
        self.plot_widget.autoRange()

        # Create scatter plot item for actual vs. predicted values
        scatter = pg.ScatterPlotItem(x=X_clean, y=Y_clean, pen=None, brush=brushes)
        self.plot_widget.addItem(scatter)

        # Retrieve the units for the target variable from the selected file
        selected_file_key = selected_files[0] if selected_files else None
        units = self.units_dict.get(selected_file_key, {})  # Dictionary of units for the selected file
        x_unit = units.get(selected_item_x, '')  # Get the unit for the target column
        y_unit = units.get(selected_item_y, '')  # Get the unit for the target column

        # Set axis labels with units
        x_unit_label = f"{selected_item_x} ({x_unit})"
        y_unit_label = f"{selected_item_y} ({y_unit})"

        self.plot_widget.setLabel('bottom', x_unit_label)  # x-axis label with unit
        self.plot_widget.setLabel('left', y_unit_label)  # y-axis label with unit

        # Create or update the ColorBarItem for the color bar
        if hasattr(self, 'well_color_bar') and self.well_color_bar is not None:
            self.well_color_bar.setLevels((labels.min(), labels.max()))
            self.plot_widget.getPlotItem().layout.removeItem(self.color_bar_label_item)
            self.well_color_bar.label = "Clusters"  # Update the color bar title with the third column name
            self.well_color_bar.setColorMap(cmap)
        else:
            color_bar = pg.ColorBarItem(values=(labels.min(), labels.max()))
            color_bar.setColorMap(cmap)
            color_bar.label = "Clusters"  # Set the color bar title to the third column name
            self.well_color_bar = color_bar
            self.plot_widget.getPlotItem().layout.addItem(self.well_color_bar, 2,
                                                          2)  # Add to the layout in the next column

        # Add vertical label for the color bar with units
        self.color_bar_label_item = pg.LabelItem(text="Clusters", angle=90,
                                                 color='k', size='10pt')
        self.plot_widget.getPlotItem().layout.addItem(self.color_bar_label_item, 2, 2, 1, 1)

        self.plot_widget.getPlotItem().layout.setContentsMargins(0, 0, 30,
                                                                 0)  # Adjust the values as needed

        self.plot_widget.autoRange()

        # Optionally, display statistics in the result layout
        result_text = f"{clustering_algorithm} Results:\n"
        result_text += f"Features: {selected_features}\n"

        result_text += f"Number of Samples: {num_valid_points}\n"

        # Silhouette Score
        silhouette_avg = silhouette_score(X_clean_copy, labels)
        result_text += f"Silhouette Score: {silhouette_avg:.2f}\n"

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(X_clean_copy, labels)
        result_text += f"Davies-Bouldin Index: {db_index:.2f}\n"

        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(X_clean_copy, labels)
        result_text += f"Calinski-Harabasz Index: {ch_index:.2f}\n"

        # Inertia (K-Means specific)
        if inertia is not None:
            result_text += f"Inertia: {inertia:.2f}\n"

        # Update the result area
        self.update_result_area(result_text, result_layout)

    def select_target_variable(self, parent_dialog, selected_files, selected_features, remaining_columns,
                               regression_type_combo, result_layout, analysis_type):
        # Create a dialog to select the target variable
        target_variable_dialog = QDialog(parent_dialog)
        target_variable_dialog.setWindowTitle("Select Target Variable")

        layout = QVBoxLayout(target_variable_dialog)

        target_variable_list = QListWidget()
        target_variable_list.setSelectionMode(QListWidget.SingleSelection)
        for col in remaining_columns:
            target_variable_list.addItem(col)
        layout.addWidget(target_variable_list)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(target_variable_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        target_variable_dialog.setLayout(layout)
        target_variable_dialog.exec()  # Use exec() to block until the dialog is closed

        # Get the selected target variable
        selected_target_items = target_variable_list.selectedItems()
        if len(selected_target_items) == 0:
            print("Please select a target variable.")
            return

        selected_target = selected_target_items[0].text()

        # Check for valid data points across selected features and target variable
        combined_df = pd.concat([self.tensor_dict[file] for file in selected_files])
        combined_df[selected_features + [selected_target]] = combined_df[selected_features + [selected_target]].apply(
            pd.to_numeric, errors='coerce')
        combined_df = combined_df.replace(-999.25, np.nan)
        valid_rows = combined_df.dropna(subset=selected_features + [selected_target])
        num_valid_points = len(valid_rows)

        if num_valid_points == 0:
            QMessageBox.warning(parent_dialog, "No Valid Data Points",
                                "No data points have values for all selected columns including the target variable.")
            return
        else:
            print(
                f"Number of data points with values for all selected columns including the target variable: {num_valid_points}")

        # Now you have selected_features and selected_target, and can proceed to the next steps
        # e.g., applying the regression methods on the selected columns
        self.data_preparation(selected_files, selected_features, selected_target, regression_type_combo, result_layout, parent_dialog, analysis_type, num_valid_points)

    def data_preparation(self, selected_files, selected_features, selected_target, regression_type_combo,
                         result_layout, parent_dialog, analysis_type, num_valid_points):
        dialog = QDialog(parent_dialog)
        dialog.setWindowTitle("Select Normalization Type")
        layout = QVBoxLayout()

        label = QLabel("Choose normalization type:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(["Min-Max Scaler", "Standard Scaler", "None"])
        layout.addWidget(combo)

        button = QPushButton("OK")
        button.setFocusPolicy(Qt.NoFocus)
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(225)
        # Disable the '?' help button on the dialog
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.adjustSize()
        if dialog.exec_() == QDialog.Accepted:
            normalization_choice = combo.currentText()
        else:
            normalization_choice = "None"

        # Load and merge data from the selected CSV files
        data_frames = [self.tensor_dict[file] for file in selected_files]
        merged_data = pd.concat(data_frames, join='inner', ignore_index=True)

        # Replace -999.25 with NaN
        merged_data.replace(-999.25, pd.NA, inplace=True)

        # Extract feature inputs and target variable
        X = merged_data[selected_features]
        y = merged_data[selected_target]

        # Convert all columns to numeric, forcing errors to NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        y = y.apply(pd.to_numeric, errors='coerce')

        # Replace negative values with NaN
        X = X.applymap(lambda x: pd.NA if x < 0 else x)
        y = y.apply(lambda x: pd.NA if x < 0 else x)

        # Drop rows with NaN values
        X = X.dropna()
        y = y.dropna()

        # Ensure X and y have the same length
        merged_cleaned_data = X.join(y, how='inner')

        # Extract the cleaned feature inputs and target variable
        X_clean = merged_cleaned_data[selected_features]
        X_clean_copy = X_clean.copy()
        y_clean = merged_cleaned_data[selected_target]

        # Normalize the feature inputs based on the user's choice
        if normalization_choice == "Min-Max Scaler":
            scaler = MinMaxScaler()
            X_clean = scaler.fit_transform(X_clean)
        elif normalization_choice == "Standard Scaler":
            scaler = StandardScaler()
            X_clean = scaler.fit_transform(X_clean)

        # Convert to numpy arrays
        X_clean = np.asarray(X_clean, dtype=float)
        y_clean = np.asarray(y_clean, dtype=float)

        self.dimensionality_reduction(parent_dialog, result_layout, X_clean, y_clean, X_clean_copy,
                                      selected_files, selected_target, selected_features, regression_type_combo, num_valid_points)

    def dimensionality_reduction(self, parent_dialog, result_layout, X_clean, y_clean, X_clean_copy,
                                 selected_files, selected_target, selected_features, regression_type_combo, num_valid_points):
        n_components = None
        # Create a new dialog to select linear or non-linear regression
        cross_plot_dialog = QDialog(parent_dialog)
        cross_plot_dialog.setWindowTitle("Dimensionality Reduction Options")

        layout = QVBoxLayout(cross_plot_dialog)

        dimensionality_reduction_type_combo = QComboBox()
        dimensionality_reduction_type_combo.addItems([
            "None", "t-SNE", "Isomap", "ICA", "Random Projection", "PCA", "Autoencoder"

        ])
        layout.addWidget(dimensionality_reduction_type_combo)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(cross_plot_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        cross_plot_dialog.setLayout(layout)
        cross_plot_dialog.adjustSize()
        cross_plot_dialog.setMinimumWidth(250)
        # Disable the '?' help button on the dialog
        cross_plot_dialog.setWindowFlags(cross_plot_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        result = cross_plot_dialog.exec()  # Use exec() to block until the dialog is closed
        if result == QDialog.Accepted:
            dimensionality_reduction_type_combo = dimensionality_reduction_type_combo.currentText()

            if dimensionality_reduction_type_combo in ["t-SNE", "Isomap", "ICA", "Random Projection", "PCA", "Autoencoder"]:

                dialog = QDialog(parent_dialog)
                dialog.setWindowTitle("Principal Components")
                layout = QVBoxLayout()

                input_field_label = QLabel()
                input_field_label.setText("Select the Number of Components:")
                layout.addWidget(input_field_label)

                input_field = QLineEdit()
                # Using QDoubleValidator to allow decimals and integers
                validator = QDoubleValidator(0, X_clean_copy.shape[1], 2, input_field)
                validator.setNotation(QDoubleValidator.StandardNotation)
                input_field.setValidator(validator)
                layout.addWidget(input_field)

                button = QPushButton("OK")
                button.clicked.connect(dialog.accept)
                layout.addWidget(button)

                dialog.setLayout(layout)
                dialog.setMinimumWidth(225)
                # Disable the '?' help button on the dialog
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.adjustSize()
                if dialog.exec_() == QDialog.Accepted:
                    n_components = float(input_field.text())
                    if n_components.is_integer():
                        # Convert to int if the number is an integer
                        n_components = int(n_components)

                    if dimensionality_reduction_type_combo == "PCA":
                        # Create PCA object to retain 90% of variance
                        pca = PCA(n_components=n_components)

                        # Fit PCA model to data
                        pca.fit(X_clean)

                        X_clean = pca.transform(X_clean)

                    elif dimensionality_reduction_type_combo == "t-SNE":
                        # Create t-SNE object
                        tsne = TSNE(n_components=n_components)
                        # Fit and transform data
                        X_clean = tsne.fit_transform(X_clean)

                    elif dimensionality_reduction_type_combo == "ICA":
                        # Create ICA object
                        ica = FastICA(n_components=n_components, max_iter=500)
                        # Fit ICA model to data
                        ica.fit(X_clean)
                        X_clean = ica.transform(X_clean)

                    elif dimensionality_reduction_type_combo == "Isomap":
                        # Create Isomap object
                        isomap = Isomap(n_components=n_components)
                        # Fit and transform data
                        X_clean = isomap.fit_transform(X_clean)

                    elif dimensionality_reduction_type_combo == "Random Projection":
                        # Create Random Projection object
                        grp = GaussianRandomProjection(n_components=n_components)
                        # Fit and transform data
                        X_clean = grp.fit_transform(X_clean)
                    elif dimensionality_reduction_type_combo == "Autoencoder":
                        # Define the input dimension based on X_queen's shape
                        input_dim = X_clean.shape[1]

                        # Split the data into training and validation sets
                        X_train, X_val = train_test_split(X_clean, test_size=0.2, random_state=42)

                        # Define the encoding dimension
                        encoding_dim = max(1, n_components)

                        # Input layer
                        input_layer = Input(shape=(input_dim,))

                        # Encoder: Dense layers
                        x = Dense(128, activation='relu')(input_layer)
                        x = BatchNormalization()(x)
                        x = Dropout(0.2)(x)

                        # Check if we can reshape it properly for Conv1D
                        if x.shape[1] == input_dim:  # Only reshape if it matches input_dim
                            x = Reshape((input_dim, 1))(x)  # Reshape for Conv1D
                            x = Conv1D(32, 3, activation='relu', padding='same')(x)
                            x = BatchNormalization()(x)
                            x = Dropout(0.2)(x)
                            x = Flatten()(x)  # Flatten to pass to dense layers
                        else:
                            # Skip Conv1D if reshape isn't possible directly from Dense
                            x = Dense(64, activation='relu')(x)

                        # Multi-Head Attention for capturing different "views" or dependencies
                        x = Reshape((1, -1))(x)  # Reshape to (batch_size, 1, features)
                        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                        attention_output = Dropout(0.2)(attention_output)
                        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)  # Residual connection
                        attention_output = Reshape((-1,))(attention_output)  # Flatten

                        # Final dense layer to reduce to the encoding dimension
                        encoded = Dense(encoding_dim, activation='relu')(attention_output)

                        # Decoder: Reverse the process
                        x = Dense(128, activation='relu')(encoded)
                        x = BatchNormalization()(x)
                        x = Dropout(0.2)(x)

                        # Adding Dense layers
                        x = Dense(64, activation='relu')(x)
                        x = BatchNormalization()(x)
                        x = Dropout(0.2)(x)

                        # Reconstructing the original input dimensions
                        decoded = Dense(input_dim, activation='sigmoid', dtype='float32')(x)

                        # Autoencoder model
                        autoencoder = Model(input_layer, decoded)

                        # Compile the model
                        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

                        # Encoder model to reduce dimensionality
                        encoder = Model(input_layer, encoded)

                        # Summary of the autoencoder
                        autoencoder.summary()

                        # Train the autoencoder with training-validation split
                        autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True,
                                        validation_data=(X_val, X_val))

                        # Use the encoder to transform the data
                        X_clean = encoder.predict(X_clean)

            else:
                pass

            print(X_clean.shape)
            print(X_clean_copy.shape)
            self.run_regression(parent_dialog, result_layout, X_clean, y_clean, X_clean_copy,
                                regression_type_combo, selected_files, selected_target, selected_features, num_valid_points, dimensionality_reduction_type_combo, n_components)

    def run_regression(self, parent_dialog, result_layout, X_clean, y_clean, X_clean_copy,
                       regression_type_combo, selected_files, selected_target, selected_features, num_valid_points, dimensionality_reduction_type_combo, n_components):

        def add_model_to_dict(model_dict, model_type, model, features, target, file_path, dimensionality_reduction_type_combo, n_components):
            """
            Adds a trained model, its metadata, and file path to the model dictionary.

            :param model_dict: The dictionary to store models and their metadata.
            :param model_type: Type of the model (e.g., "RandomForest", "LinearRegression").
            :param model: The trained model object.
            :param features: List of feature input names used for training the model.
            :param target: The target variable name the model predicts.
            :param file_path: Path to the saved model file.
            """
            model_metadata = {
                "model": model,
                "features": features,
                "target": target,
                "file_path": file_path,
                "dim_type": dimensionality_reduction_type_combo,
                "n_components": n_components

            }

            combined_key = f"{model_type}_{dimensionality_reduction_type_combo}"

            if combined_key not in model_dict:
                model_dict[combined_key] = []

            model_dict[combined_key].append(model_metadata)

        dialog = QDialog(parent_dialog)
        dialog.setWindowTitle("Train test split")
        layout = QVBoxLayout()

        label = QLabel("Choose the train test split")
        layout.addWidget(label)

        # Add a spin box for integer input
        spinBox = QSpinBox()
        spinBox.setMinimum(1)
        spinBox.setMaximum(99)
        spinBox.setValue(20)
        layout.addWidget(spinBox)

        button = QPushButton("OK")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(225)
        # Disable the '?' help button on the dialog
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.adjustSize()
        if dialog.exec_() == QDialog.Accepted:
            train_test = spinBox.value()

        # Split into training and validation sets (80% train, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=train_test / 100, random_state=42)

        # Function to create a simple Keras model for regression
        def create_keras_model():
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            return model

        if regression_type_combo in ["Random Forest Regressor",  "Gradient Boosting Regressor", "AdaBoost Regressor",
                                     "XGBoost Regressor", "CatBoost Regressor", "Keras Regressor", "ResNet", "TableNet",
                                     "TCN", "LightGBM Regressor", "NGBoost Regressor"]:

            dialog = QDialog(parent_dialog)
            dialog.setWindowTitle("Epochs")
            layout = QVBoxLayout()

            label = QLabel("Choose the number of epochs")
            layout.addWidget(label)

            # Add a spin box for integer input
            spinBox = QSpinBox()
            spinBox.setMinimum(1)  # Set minimum number of epochs to 1
            spinBox.setMaximum(1000)  # Set maximum number of epochs to 1000
            spinBox.setValue(20)
            layout.addWidget(spinBox)

            button = QPushButton("OK")
            button.clicked.connect(dialog.accept)
            layout.addWidget(button)

            dialog.setLayout(layout)
            dialog.setMinimumWidth(225)
            # Disable the '?' help button on the dialog
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.adjustSize()
            if dialog.exec_() == QDialog.Accepted:
                epochs = spinBox.value()

        # Initialize the model based on the regression type
        if regression_type_combo == "Ridge Regression":
            model = Ridge()

        elif regression_type_combo == "Lasso Regression":
            model = Lasso()

        elif regression_type_combo == "Bayesian Ridge Regression":
            model = BayesianRidge()

        elif regression_type_combo == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=epochs)  # Using epochs as number of estimators

        elif regression_type_combo == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(n_estimators=epochs)  # Using epochs as number of estimators

        elif regression_type_combo == "AdaBoost Regressor":
            model = AdaBoostRegressor(n_estimators=epochs)  # Using epochs as number of estimators

        elif regression_type_combo == "Support Vector Regressor":
            model = SVR()

        elif regression_type_combo == "K-Nearest Neighbors Regressor":
            model = KNeighborsRegressor()

        elif regression_type_combo == "Decision Tree Regressor":
            model = DecisionTreeRegressor()

        elif regression_type_combo == "Gaussian Process Regressor":
            model = GaussianProcessRegressor()

        elif regression_type_combo == "XGBoost Regressor":
            model = XGBRegressor(n_estimators=epochs)  # Using epochs as number of estimators

        elif regression_type_combo == "CatBoost Regressor":
            model = CatBoostRegressor(iterations=epochs, verbose=0)  # Using epochs as number of iterations

        elif regression_type_combo == "Keras Regressor":
            model = KerasRegressor(build_fn=create_keras_model, epochs=epochs, batch_size=10, verbose=0)

        elif regression_type_combo == "ResNet":
            input_dim = X_train.shape[1]
            model = ResNet1D(input_dim=input_dim)

        elif regression_type_combo == "TableNet":
            model = TabNetRegressor()

        elif regression_type_combo == "TCN":
            num_channels = [25, 50, 100]  # Example configuration
            model = TCN(num_inputs=1, num_channels=num_channels)

        elif regression_type_combo == "LightGBM Regressor":
            model = LGBMRegressor(n_estimators=100 * epochs)

        elif regression_type_combo == "NGBoost Regressor":
            model = NGBRegressor(n_estimators=100 * epochs)  # Using epochs as number of estimators

        else:
            raise ValueError(f"Unknown regression type: {regression_type_combo}")

        # Fit the model
        if regression_type_combo == "Keras Regressor":
            QApplication.setOverrideCursor(self.custom_cursor)
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
        elif regression_type_combo == "ResNet":
            QApplication.setOverrideCursor(self.custom_cursor)
            # Check if GPU is available and move model to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Define the loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scaler = GradScaler()

            # Convert training and validation data to PyTorch tensors and move to GPU if available
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)  # Reshape y only for ResNet
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)  # Reshape y only for ResNet

            # Create DataLoader for mini-batches
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Training loop with mini-batches and mixed precision
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    # Forward pass with mixed precision
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)

                    # Backward pass with mixed precision
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(device, non_blocking=True),\
                            y_val_batch.to(device, non_blocking=True)
                        with autocast():
                            val_outputs = model(X_val_batch)
                            val_loss += criterion(val_outputs, y_val_batch).item()

                avg_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                # Print training and validation loss
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        elif regression_type_combo == "TableNet":
            QApplication.setOverrideCursor(self.custom_cursor)
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_name=['val'],
                eval_metric=['rmse'],
                max_epochs=epochs,
                patience=10,
                batch_size=64,
                virtual_batch_size=32,
                num_workers=0,
                drop_last=False
            )

        elif regression_type_combo == "TCN":
            # GPU handling for TCN
            QApplication.setOverrideCursor(self.custom_cursor)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scaler = GradScaler()

            # TCN expects a 1D input in the format (batch_size, channels, length)
            X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], 1, -1), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val.reshape(X_val.shape[0], 1, -1), dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(device, non_blocking=True),\
                            y_val_batch.to(device, non_blocking=True)
                        with autocast():
                            val_outputs = model(X_val_batch)
                            val_loss += criterion(val_outputs, y_val_batch).item()

                avg_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        elif regression_type_combo == "LightGBM Regressor":
            QApplication.setOverrideCursor(self.custom_cursor)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        elif regression_type_combo in ["NGBoost Regressor"]:
            QApplication.setOverrideCursor(self.custom_cursor)
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Predict using the fitted model
        if regression_type_combo == "ResNet":
            # Make predictions
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_tensor.to(device)).cpu().numpy()  # Move predictions back to CPU for compatibility
                y_pred = np.array(y_pred).squeeze()
            QApplication.restoreOverrideCursor()
            torch.cuda.empty_cache()
        elif regression_type_combo == "TableNet":
            y_pred = model.predict(X_val)
            y_pred = np.array(y_pred).squeeze()
            y_val = np.array(y_val).squeeze()
            QApplication.restoreOverrideCursor()
            torch.cuda.empty_cache()
        elif regression_type_combo == "TCN":
            # Make predictions
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_tensor.to(device)).cpu().numpy()  # Move predictions back to CPU for compatibility
                y_pred = np.array(y_pred).squeeze()
            QApplication.restoreOverrideCursor()
            torch.cuda.empty_cache()
        else:
            y_pred = model.predict(X_val)
            QApplication.restoreOverrideCursor()

        if regression_type_combo in ["Keras Regressor"]:
            # Check if it's a KerasRegressor and extract the underlying model
            # Add models to the dictionary
            file_dialog = QFileDialog()
            file_directory = file_dialog.getExistingDirectory(None, "Select Directory to Save Model")

            if hasattr(model, 'model'):
                model = model.model  # Extract the actual Keras model
            file_path = f"{file_directory}/{regression_type_combo}model{selected_files, selected_target, selected_features}.h5"
            model.save(file_path)

            # Adding model to dictionary with the file path
            add_model_to_dict(self.model_dict, regression_type_combo, model, selected_features, selected_target,
                              file_path, dimensionality_reduction_type_combo, n_components)

        elif regression_type_combo in ["TCN", "ResNet", "TableNet"]:
            # Add models to the dictionary
            file_dialog = QFileDialog()
            file_directory = file_dialog.getExistingDirectory(None, "Select Directory to Save Model")

            file_path = f"{file_directory}/{regression_type_combo}model{selected_files, selected_target, selected_features}.pth"
            torch.save(model, file_path)  # Save the whole PyTorch model
            # Adding model to dictionary with the file path
            add_model_to_dict(self.model_dict, regression_type_combo, model, selected_features, selected_target,
                              file_path, dimensionality_reduction_type_combo, n_components)
        else:  # Assume Sklearn or similar

            # Ask the user if they want to save the model
            reply = QMessageBox.question(self, 'Save Model', 'Do you want to save the model?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                file_dialog = QFileDialog()
                file_directory = file_dialog.getExistingDirectory(None, "Select Directory to Save Model")
                file_path = f"{file_directory}/{regression_type_combo}model{selected_files, selected_target, selected_features}.pkl"
                joblib.dump(model, file_path)
            else:
                file_path = None
            # Adding model to dictionary with the file path
            add_model_to_dict(self.model_dict, regression_type_combo, model, selected_features, selected_target,
                              file_path, dimensionality_reduction_type_combo, n_components)

        # Pass data to the plotting function
        self.plot_results(y_val, y_pred, regression_type_combo, result_layout, selected_files, selected_target, selected_features, parent_dialog, X_clean_copy, train_test, num_valid_points)

    def plot_results(self, y_actual, y_pred, regression_type, result_layout, selected_files, selected_target, selected_features, parent_dialog, X_clean_copy, train_test, num_valid_points):
        """
        Plot the predicted values against the actual values for regression analysis.
        """

        # Create a dialog to select the target variable
        target_variable_dialog = QDialog(parent_dialog)
        target_variable_dialog.setWindowTitle("Select Color-coded Feature")

        layout = QVBoxLayout(target_variable_dialog)

        target_variable_list = QListWidget()
        target_variable_list.setSelectionMode(QListWidget.SingleSelection)
        for col in selected_features:
            target_variable_list.addItem(col)
        layout.addWidget(target_variable_list)

        select_button = QPushButton("OK")
        select_button.setFocusPolicy(Qt.NoFocus)
        select_button.clicked.connect(target_variable_dialog.accept)  # Close the current dialog
        layout.addWidget(select_button)

        target_variable_dialog.setLayout(layout)
        target_variable_dialog.exec()  # Use exec() to block until the dialog is closed

        # Get the selected target variable
        selected_target_items = target_variable_list.selectedItems()
        if len(selected_target_items) == 0:
            print("Please select a target variable.")
            return

        selected_item = selected_target_items[0].text()

        # Extract the cleaned feature inputs and target variable
        X_clean = X_clean_copy[selected_item]

        # Convert to numpy arrays
        X_clean = np.asarray(X_clean, dtype=float)

        # Split into training and validation sets (80% train, 20% validation)
        X_train, c_data = train_test_split(X_clean, test_size=train_test / 100, random_state=42)

        # Normalize c_data for color mapping
        norm_c_data = (c_data - c_data.min()) / (c_data.max() - c_data.min())

        self.well_color_data = norm_c_data

        # Use a colormap with sufficient distinct colors
        cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')
        colors = cmap.map(self.well_color_data, mode='qcolor')

        # Clear the previous plot and reset the view
        self.plot_widget.clear()
        self.plot_widget.autoRange()

        # Create scatter plot item for actual vs. predicted values
        scatter = pg.ScatterPlotItem(x=y_actual, y=y_pred, pen=None, brush=colors)
        self.plot_widget.addItem(scatter)

        # Retrieve the units for the target variable from the selected file
        selected_file_key = selected_files[0] if selected_files else None
        units = self.units_dict.get(selected_file_key, {})  # Dictionary of units for the selected file
        target_unit = units.get(selected_target, '')  # Get the unit for the target column

        # Set axis labels with units
        x_unit_label = f"Actual {selected_target} ({target_unit})"
        y_unit_label = f"Predicted {selected_target} ({target_unit})"

        self.plot_widget.setLabel('bottom', x_unit_label)  # x-axis label with unit
        self.plot_widget.setLabel('left', y_unit_label)  # y-axis label with unit

        # Optionally, you can add a line y=x to indicate perfect predictions
        diag_line = pg.PlotCurveItem(x=[min(y_actual), max(y_actual)], y=[min(y_actual), max(y_actual)],
                                     pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addItem(diag_line)

        # Create or update the ColorBarItem for the color bar
        if hasattr(self, 'well_color_bar') and self.well_color_bar is not None:
            self.well_color_bar.setLevels((c_data.min(), c_data.max()))
            self.plot_widget.getPlotItem().layout.removeItem(self.color_bar_label_item)
            self.well_color_bar.label = selected_item  # Update the color bar title with the third column name
            self.well_color_bar.setColorMap(cmap)
        else:
            color_bar = pg.ColorBarItem(values=(c_data.min(), c_data.max()))
            color_bar.setColorMap(cmap)
            color_bar.label = selected_item  # Set the color bar title to the third column name
            self.well_color_bar = color_bar
            self.plot_widget.getPlotItem().layout.addItem(self.well_color_bar, 2,
                                                          2)  # Add to the layout in the next column

        # Add vertical label for the color bar with units
        color_bar_unit_label = f"({units.get(selected_item, '')})"
        self.color_bar_label_item = pg.LabelItem(text=f"{selected_item} {color_bar_unit_label}", angle=90,
                                                 color='k', size='10pt')
        self.plot_widget.getPlotItem().layout.addItem(self.color_bar_label_item, 2, 2, 1, 1)

        self.plot_widget.getPlotItem().layout.setContentsMargins(0, 0, 30,
                                                                 0)  # Adjust the values as needed

        self.plot_widget.autoRange()

        def robust_mape(y_true, y_pred, epsilon=1e-10, max_error=None):
            """
            Calculate the Mean Absolute Percentage Error (MAPE) with robust handling of edge cases.

            Parameters:
            - y_true: array-like, true values.
            - y_pred: array-like, predicted values.
            - epsilon: small value to replace zeros or very small values in y_true to prevent division by zero (default: 1e-10).
            - max_error: optional, maximum percentage error to clip extreme values (e.g., 1000 for capping at 1000%).

            Returns:
            - A string with the computed MAPE as a percentage or an error message if calculation is not possible.
            """
            try:
                # Convert inputs to numpy arrays for element-wise operations
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)

                # Check for NaNs or Infs in the inputs
                if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
                    return "Error: Input arrays contain NaNs."
                if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
                    return "Error: Input arrays contain Infs."

                # Replace zeros or very small values in y_true with epsilon to prevent division by zero
                y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)

                # Compute the absolute percentage errors
                percentage_errors = np.abs((y_true - y_pred) / y_true_safe) * 100

                # Optionally clip extreme percentage errors if max_error is provided
                if max_error is not None:
                    percentage_errors = np.clip(percentage_errors, 0, max_error)

                # Calculate the mean of the absolute percentage errors
                mape = np.mean(percentage_errors)

                return f"MAPE: {mape:.2f}%"

            except Exception as e:
                # Handle any unexpected exceptions and return the error message
                return f"Error: {str(e)}"

        # Optionally, display statistics in the result layout
        result_text = f"{regression_type} Results:\n"
        result_text += f"Number of Samples: {num_valid_points}\n"
        result_text += f"Features: {selected_features}\n"
        result_text += f"Correlation Coefficient: {pearsonr(y_actual, y_pred)[0]:.2f}\n"
        result_text += f"R-squared Score: {r2_score(y_actual, y_pred):.2f}\n"
        result_text += f"MSE: {mean_squared_error(y_actual, y_pred):.6f}\n"
        result_text += f"RMSE: {root_mean_squared_error(y_actual, y_pred):.6f}\n"
        result_text += f"MAE: {mean_absolute_error(y_actual, y_pred):.6f}\n"
        result_text += robust_mape(y_actual, y_pred, epsilon=1e-10, max_error=1000)

        # Update the result area
        self.update_result_area(result_text, result_layout)

    def plot_cross_plot(self, selected_files, common_columns_list, regression_type_combo, result_layout, parent_dialog, num_valid_points):

        # Plot the cross plot based on selected columns and regression type
        selected_columns = [item.text() for item in common_columns_list.selectedItems()]

        if len(selected_columns) < 2:
            print("Please select at least 2 columns.")
            return

        # Assume there's only one selected file for simplicity in retrieving units
        selected_file_key = selected_files[0] if selected_files else None

        # Retrieve the units dictionary for the selected file
        units = self.units_dict.get(selected_file_key, {})

        # Initialize x_data, y_data, and c_data as None
        x_data, y_data, c_data = None, None, None

        if len(selected_files) == 1:
            # Only one file selected
            df = self.tensor_dict.get(selected_files[0])
            if df is not None:
                x_data = df[selected_columns[0]]
                y_data = df[selected_columns[1]]

                if len(selected_columns) == 3:
                    c_data = df[selected_columns[2]]
        else:
            # Multiple files selected
            data_frames = [self.tensor_dict.get(file) for file in selected_files if
                           self.tensor_dict.get(file) is not None]
            if not data_frames:
                print("Selected columns not found in the DataFrames.")
                return

            merged_data = data_frames[0]
            for df in data_frames[1:]:
                merged_data = pd.merge(merged_data, df, on=selected_columns, how='outer')

            x_data = merged_data[selected_columns[0]]
            y_data = merged_data[selected_columns[1]]

            if len(selected_columns) == 3:
                c_data = merged_data[selected_columns[2]]

        # Convert data to numeric type, ignoring non-numeric data and treating negative values as NaN
        x_data = pd.to_numeric(x_data, errors='coerce')
        y_data = pd.to_numeric(y_data, errors='coerce')

        # Replace negative values with NaN
        x_data = x_data.apply(lambda x: pd.NA if x < 0 else x)
        y_data = y_data.apply(lambda x: pd.NA if x < 0 else x)

        # Filter out NaN values
        mask = x_data.notna() & y_data.notna()
        if c_data is not None:
            c_data = pd.to_numeric(c_data, errors='coerce')
            c_data = c_data.apply(lambda x: pd.NA if x < 0 else x)
            mask &= c_data.notna()

        x_data = x_data[mask]
        y_data = y_data[mask]
        if c_data is not None:
            c_data = c_data[mask]

        # Ensure all data is numeric and finite
        x_data = x_data.astype(float)
        y_data = y_data.astype(float)
        if c_data is not None:
            c_data = c_data.astype(float)

        self.plot_widget.clear()
        self.plot_widget.autoRange()

        if c_data is not None:
            # Normalize c_data for color mapping
            self.well_color_data = (c_data - c_data.min()) / (c_data.max() - c_data.min())

            # Use a colormap with sufficient distinct colors
            cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')
            colors = cmap.map(self.well_color_data, mode='qcolor')

            scatter = pg.ScatterPlotItem(x=x_data, y=y_data, pen=None, brush=colors)
            self.plot_widget.addItem(scatter)

            # Create or update the ColorBarItem for the color bar
            if hasattr(self, 'well_color_bar') and self.well_color_bar is not None:
                self.well_color_bar.setLevels((c_data.min(), c_data.max()))
                self.plot_widget.getPlotItem().layout.removeItem(self.color_bar_label_item)
                self.well_color_bar.label = selected_columns[2]  # Update the color bar title with the third column name
                self.well_color_bar.setColorMap(cmap)
            else:
                color_bar = pg.ColorBarItem(values=(c_data.min(), c_data.max()))
                color_bar.setColorMap(cmap)
                color_bar.label = selected_columns[2]  # Set the color bar title to the third column name
                self.well_color_bar = color_bar
                self.plot_widget.getPlotItem().layout.addItem(self.well_color_bar, 2,
                                                              2)  # Add to the layout in the next column

            # Add vertical label for the color bar with units
            color_bar_unit_label = f"({units.get(selected_columns[2], '')})"
            self.color_bar_label_item = pg.LabelItem(text=f"{selected_columns[2]} {color_bar_unit_label}", angle=90,
                                                     color='k', size='10pt')
            self.plot_widget.getPlotItem().layout.addItem(self.color_bar_label_item, 2, 2, 1, 1)

            self.plot_widget.getPlotItem().layout.setContentsMargins(0, 0, 30, 0)  # Adjust the values as needed

        else:
            cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')

            color_value = 1

            # Get the QColor object for the specific value from the colormap
            color = cmap.map(color_value, mode='qcolor')

            # Create the ScatterPlotItem with the chosen brush color
            scatter = pg.ScatterPlotItem(x=x_data, y=y_data, pen=None, symbol='o', brush=color)
            self.plot_widget.addItem(scatter)
            self.well_color_data = np.ones(len(x_data), dtype=int)
            # Remove the color bar if it exists
            if hasattr(self, 'well_color_bar') and self.well_color_bar is not None:
                self.plot_widget.getPlotItem().layout.removeItem(self.color_bar_label_item)
                self.plot_widget.getPlotItem().layout.removeItem(self.well_color_bar)
                self.well_color_bar = None
                self.color_bar_label_item = None

        # Set axis labels with units
        x_unit_label = f"({units.get(selected_columns[0], '')})"
        y_unit_label = f"({units.get(selected_columns[1], '')})"
        self.plot_widget.setLabel('bottom', f"{selected_columns[0]} {x_unit_label}")  # x-axis label with unit
        self.plot_widget.setLabel('left', f"{selected_columns[1]} {y_unit_label}")  # y-axis label with unit

        # Assuming self.regression_type_combo is accessible or pass it as a parameter
        regression_type = regression_type_combo.currentText()
        if regression_type == "Linear Regression":
            self.perform_linear_fit(x_data, y_data, result_layout, num_valid_points)
        else:
            self.perform_nonlinear_fit(x_data, y_data, result_layout, num_valid_points)

    def perform_linear_fit(self, x_data, y_data, result_layout, num_valid_points):
        # Reshape the data to fit the model requirements
        x_data_reshaped = x_data.values.reshape(-1, 1)
        y_data_reshaped = y_data.values.reshape(-1, 1)

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(x_data_reshaped, y_data_reshaped)

        # Make predictions
        y_fit = model.predict(x_data_reshaped).flatten()  # Flatten to 1D

        # Calculate correlation coefficients
        pearson_corr, _ = pearsonr(x_data.values.flatten(), y_data.values.flatten())
        spearman_corr, _ = spearmanr(x_data.values.flatten(), y_data.values.flatten())
        r_squared = model.score(x_data_reshaped, y_data_reshaped)

        # Prepare the result text
        result_text = f"y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}\n"
        result_text += f"Number of Samples: {num_valid_points}\n"
        result_text += f"Intercept: {model.intercept_[0]:.2f}\n"
        result_text += f"Pearson Correlation: {pearson_corr:.2f}\n"
        result_text += f"Spearman Correlation: {spearman_corr:.2f}\n"
        result_text += f"R-squared Score: {r_squared:.2f}\n"

        # Plot the fitted line using PyQtGraph
        cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')

        color_value = 0.5

        # Get the QColor object for the specific value from the colormap
        color = cmap.map(color_value, mode='qcolor')

        line = pg.PlotCurveItem(x_data.values, y_fit, pen=color)  # Ensure x_data is 1D
        self.plot_widget.autoRange()
        self.plot_widget.addItem(line)

        # Update the result area
        self.update_result_area(result_text, result_layout)

    def perform_nonlinear_fit(self, x_data, y_data, result_layout, num_valid_points):
        # Reshape the data to fit the model requirements
        x_data_reshaped = x_data.values.reshape(-1, 1)
        y_data_reshaped = y_data.values.reshape(-1, 1)

        # Transform the data into polynomial features
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x_data_reshaped)

        # Fit the polynomial regression model
        model = LinearRegression()
        model.fit(x_poly, y_data_reshaped)

        # Make predictions
        y_fit = model.predict(x_poly).flatten()  # Flatten to 1D

        # Generate a dense range of x values for a smooth curve
        x_dense = np.linspace(x_data.min(), x_data.max(), 1000).reshape(-1, 1)
        x_dense_poly = poly.transform(x_dense)
        y_dense_fit = model.predict(x_dense_poly).flatten()

        # Calculate correlation coefficients
        pearson_corr, _ = pearsonr(y_data.values.flatten(), y_fit)
        spearman_corr, _ = spearmanr(y_data.values.flatten(), y_fit)
        r_squared = model.score(x_poly, y_data_reshaped)

        # Prepare the result text
        result_text = f"y = {model.coef_[0][2]:.2f}x^2 + {model.coef_[0][1]:.2f}x + {model.intercept_[0]:.2f}\n"
        result_text += f"Number of Samples: {num_valid_points}\n"
        result_text += f"Intercept: {model.intercept_[0]:.2f}\n"
        result_text += f"Pearson Correlation: {pearson_corr:.2f}\n"
        result_text += f"Spearman Correlation: {spearman_corr:.2f}\n"
        result_text += f"R-squared Score: {r_squared:.2f}\n"

        # Plot the smooth fitted curve using PyQtGraph
        # Plot the fitted line using PyQtGraph
        cmap = pg.colormap.get(self.well_color_mapping, source='matplotlib')

        color_value = 0.5

        # Get the QColor object for the specific value from the colormap
        color = cmap.map(color_value, mode='qcolor')

        curve = pg.PlotCurveItem(x_dense.flatten(), y_dense_fit, pen=color)  # Ensure x_dense is flattened
        self.plot_widget.autoRange()
        self.plot_widget.addItem(curve)

        # Update the result area
        self.update_result_area(result_text, result_layout)

    def update_result_area(self, text, result_layout):
        # Clear the previous results
        for i in reversed(range(result_layout.count())):
            result_layout.itemAt(i).widget().setParent(None)

        # Add the new result
        label = QLabel(text)

        # Replace '^' with HTML superscript tags for proper formatting
        formatted_text = text.replace("^2", "<sup>2</sup>").replace("^", "<sup>")
        label.setText(f"<p style='white-space: pre;'>{formatted_text}</p>")

        # Set font
        font = QFont('Times New Roman',
                     10)  # You can change 'Times New Roman' to your preferred font and '14' to your preferred size
        label.setFont(font)

        # Set stylesheet
        label.setStyleSheet("""
        QLabel {
        color: #555;              /* Text color */
        background-color: #ddd;   /* Background color */
        border: 2px solid #ccc;   /* Border with 2px solid line and color */
        padding: 4px;             /* Padding around text */
        border-radius: 4px;       /* Rounded corners */
        }
        """)

        # Set alignment
        label.setAlignment(Qt.AlignCenter)  # Align the text to the center
        result_layout.addWidget(label)

    def instantaneous_phase(self):
        if self.inst_phase_radians.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Convert the instantaneous phase from radians to degrees
                self.tensor_data = np.degrees(self.inst_phase_radians[self.file_name])
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Phase", self.tensor_data)
                print('Instantaneous phase calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Convert the instantaneous phase from radians to degrees
                self.tensor_data = np.degrees(self.inst_phase_radians[self.file_name])
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Phase", self.tensor_data)
                print('Instantaneous phase calculation done')
                QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Convert the instantaneous phase from radians to degrees
            self.tensor_data = np.degrees(self.inst_phase_radians[self.file_name])
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Inst. Phase", self.tensor_data)
            print('Instantaneous phase calculation done')
            QApplication.restoreOverrideCursor()

    def instantaneous_frequency(self):
        if self.inst_phase_radians.get(self.file_name) is not None:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Unwrap the phase to correct for discontinuities
            inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

            sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

            # Calculate the time derivative of the unwrapped phase
            inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

            # Convert the instantaneous frequency from radians per second to Hertz
            inst_freq_hz = inst_freq / (2 * np.pi)

            # Check for negative frequencies and set them to zero if necessary
            inst_freq_hz[inst_freq_hz < 0] = 0

            self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)), 'edge')

            self.tensor_data = self.inst_freq_hz_padded[self.file_name]
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Inst. Frequency", self.tensor_data)
            print('Instantaneous frequency calculation done')
            QApplication.restoreOverrideCursor()
        else:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                self.tensor_data = self.inst_freq_hz_padded[self.file_name]
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Frequency", self.tensor_data)
                print('Instantaneous frequency calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                self.tensor_data = self.inst_freq_hz_padded[self.file_name]
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Frequency", self.tensor_data)
                print('Instantaneous frequency calculation done')
                QApplication.restoreOverrideCursor()

    def cosine_of_phase(self):
        if self.inst_phase_radians.get(self.file_name) is not None:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Unwrap the phase to correct for discontinuities
            inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

            cosine_of_phase = np.cos(inst_phase_unwrapped)

            self.tensor_data = cosine_of_phase
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Cos Phase", self.tensor_data)
            print('Cosine of Phase calculation done')
            QApplication.restoreOverrideCursor()
        else:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                cosine_of_phase = np.cos(inst_phase_unwrapped)

                self.tensor_data = cosine_of_phase
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Cos Phase", self.tensor_data)
                print('Cosine of Phase calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                cosine_of_phase = np.cos(inst_phase_unwrapped)

                self.tensor_data = cosine_of_phase
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Cos Phase", self.tensor_data)
                print('Cosine of Phase calculation done')
                QApplication.restoreOverrideCursor()

    def envelope(self):
        if self.analytic_signal.get(self.file_name) is None:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Compute the analytic signal along the amplitude axis (last axis)
            self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
            self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
            self.tensor_data = self.envelope_dict[self.file_name]
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Envelope", self.tensor_data)
            print('Envelope calculation done')
            QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(self.custom_cursor)
            self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
            self.tensor_data = self.envelope_dict[self.file_name]
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Envelope", self.tensor_data)
            print('Envelope calculation done')
            QApplication.restoreOverrideCursor()

    def sweetness(self):
        if self.inst_freq_hz_padded.get(self.file_name) is None and self.envelope_dict.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])

                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()

        elif self.inst_freq_hz_padded.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()

        elif self.envelope_dict.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])

                # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
                inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

                # Calculate Sweetness
                sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

                self.tensor_data = sweetness
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
                print('Sweetness calculation done')
                QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Ensure that the instantaneous frequency is not less than 1 Hz for numerical stability
            inst_freq_hz_threshold = np.maximum(self.inst_freq_hz_padded[self.file_name], 1)

            # Calculate Sweetness
            sweetness = self.envelope_dict[self.file_name] / np.sqrt(inst_freq_hz_threshold)

            self.tensor_data = sweetness
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Sweetness", self.tensor_data)
            print('Sweetness calculation done')
            QApplication.restoreOverrideCursor()

    def inst_bandwidth(self):
        if self.envelope_dict.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                envelope_copy = np.copy(self.envelope_dict[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                time_derivative_envelope_per_2pi = time_derivative_envelope_padded / (2 * np.pi)
                time_derivative_envelope_per_hz = time_derivative_envelope_per_2pi / envelope_copy
                data = np.abs(time_derivative_envelope_per_hz)
                sampling_interval = self.get_sampling_interval_from_file_name() / 1000
                data_final = data / sampling_interval

                data_final[data_final > 125] = 125

                self.inst_bandwidth_dict[self.file_name] = data_final

                self.tensor_data = self.inst_bandwidth_dict[self.file_name]
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Bandwidth", self.tensor_data)
                print('Inst. Bandwidth done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                envelope_copy = np.copy(self.envelope_dict[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                time_derivative_envelope_per_2pi = time_derivative_envelope_padded / (2 * np.pi)
                time_derivative_envelope_per_hz = time_derivative_envelope_per_2pi / envelope_copy
                data = np.abs(time_derivative_envelope_per_hz)
                sampling_interval = self.get_sampling_interval_from_file_name() / 1000
                data_final = data / sampling_interval

                data_final[data_final > 125] = 125

                self.inst_bandwidth_dict[self.file_name] = data_final

                self.tensor_data = self.inst_bandwidth_dict[self.file_name]
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Inst. Bandwidth", self.tensor_data)
                print('Inst. Bandwidth done')
                QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(self.custom_cursor)
            envelope_copy = np.copy(self.envelope_dict[self.file_name])
            time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
            time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                     'edge')
            # Calculate the Instantaneous Bandwidth
            time_derivative_envelope_per_2pi = time_derivative_envelope_padded / (2 * np.pi)
            time_derivative_envelope_per_hz = time_derivative_envelope_per_2pi / envelope_copy
            data = np.abs(time_derivative_envelope_per_hz)
            sampling_interval = self.get_sampling_interval_from_file_name() / 1000
            data_final = data / sampling_interval

            data_final[data_final > 125] = 125

            self.inst_bandwidth_dict[self.file_name] = data_final

            self.tensor_data = self.inst_bandwidth_dict[self.file_name]
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Inst. Bandwidth", self.tensor_data)
            print('Inst. Bandwidth done')
            QApplication.restoreOverrideCursor()

    def dominant_frequency(self):
        if self.inst_freq_hz_padded.get(self.file_name) is None and self.inst_bandwidth_dict.get(
                self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000
                time_derivative_envelope_padded_time = time_derivative_envelope_padded * sampling_interval_seconds
                self.inst_bandwidth_dict[self.file_name] = np.abs(time_derivative_envelope_padded_time)

                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')
                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000
                time_derivative_envelope_padded_time = time_derivative_envelope_padded * sampling_interval_seconds
                self.inst_bandwidth_dict[self.file_name] = np.abs(time_derivative_envelope_padded_time)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')

                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()

        elif self.inst_freq_hz_padded.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')
                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)

                # Unwrap the phase to correct for discontinuities
                inst_phase_unwrapped = np.unwrap(self.inst_phase_radians[self.file_name], axis=0)

                # Calculate the instantaneous frequency
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000

                # Calculate the time derivative of the unwrapped phase
                inst_freq = np.diff(inst_phase_unwrapped, axis=0) / sampling_interval_seconds

                # Convert the instantaneous frequency from radians per second to Hertz
                inst_freq_hz = inst_freq / (2 * np.pi)

                # Check for negative frequencies and set them to zero if necessary
                inst_freq_hz[inst_freq_hz < 0] = 0

                self.inst_freq_hz_padded[self.file_name] = np.pad(inst_freq_hz, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                                  'edge')
                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()

        elif self.inst_bandwidth_dict.get(self.file_name) is None:
            if self.analytic_signal.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Compute the analytic signal along the amplitude axis (last axis)
                self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)

                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000
                time_derivative_envelope_padded_time = time_derivative_envelope_padded * sampling_interval_seconds
                self.inst_bandwidth_dict[self.file_name] = np.abs(time_derivative_envelope_padded_time)

                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                time_derivative_envelope = np.diff(self.envelope_dict[self.file_name], axis=0)
                time_derivative_envelope_padded = np.pad(time_derivative_envelope, ((1, 0), (0, 0), (0, 0), (0, 0)),
                                                         'edge')
                # Calculate the Instantaneous Bandwidth
                sampling_interval_seconds = self.get_sampling_interval_from_file_name() / 1000
                time_derivative_envelope_padded_time = time_derivative_envelope_padded * sampling_interval_seconds
                self.inst_bandwidth_dict[self.file_name] = np.abs(time_derivative_envelope_padded_time)

                # Calculate the Dominant Frequency
                dominant_frequency = np.sqrt(
                    self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

                dominant_frequency[dominant_frequency > 125] = 125

                reverted_intensity = dominant_frequency

                self.tensor_data = reverted_intensity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
                print('Dominant Frequency calculation done')
                QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Calculate the Dominant Frequency
            dominant_frequency = np.sqrt(
                self.inst_freq_hz_padded[self.file_name] ** 2 + self.inst_bandwidth_dict[self.file_name] ** 2)

            dominant_frequency[dominant_frequency > 125] = 125

            reverted_intensity = dominant_frequency

            self.tensor_data = reverted_intensity
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_Dominant Frequency", self.tensor_data)
            print('Dominant Frequency calculation done')
            QApplication.restoreOverrideCursor()

    def apparent_polarity(self):
        if self.analytic_signal.get(self.file_name) is None:
            QApplication.setOverrideCursor(self.custom_cursor)
            # Compute the analytic signal along the amplitude axis (last axis)
            self.analytic_signal[self.file_name] = hilbert(self.tensor_data, axis=0)
            # Calculate the instantaneous phase in radians
            self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

            self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
            self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
            # Find local amplitude maxima (peaks) and minima (troughs)
            # Find indices of local maxima along the time axis
            maxima_indices = argrelextrema(self.envelope_dict[self.file_name], np.greater, axis=0)
            # Find indices of local minima along the time axis
            minima_indices = argrelextrema(self.envelope_dict[self.file_name], np.less, axis=0)
            # Calculate apparent polarity at local amplitude extrema
            # Polarity is positive at peaks and negative at troughs
            apparent_polarity = np.zeros_like(self.envelope_dict[self.file_name])
            apparent_polarity[maxima_indices] = np.cos(self.inst_phase_radians[self.file_name][maxima_indices])
            apparent_polarity[minima_indices] = -np.cos(self.inst_phase_radians[self.file_name][minima_indices])
            self.tensor_data = apparent_polarity
            # Store the modified tensor with an informative key
            self.add_tensor(f"{self.file_name}_App. Polarity", self.tensor_data)
            print('Apparent Polarity calculation done')
            QApplication.restoreOverrideCursor()
        else:
            if self.inst_phase_radians.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Calculate the instantaneous phase in radians
                self.inst_phase_radians[self.file_name] = np.angle(self.analytic_signal[self.file_name])

                self.inst_phase_radians[self.file_name] = self.inst_phase_radians[self.file_name].astype(np.float32)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                # Find local amplitude maxima (peaks) and minima (troughs)
                # Find indices of local maxima along the time axis
                maxima_indices = argrelextrema(self.envelope_dict[self.file_name], np.greater, axis=0)
                # Find indices of local minima along the time axis
                minima_indices = argrelextrema(self.envelope_dict[self.file_name], np.less, axis=0)
                # Calculate apparent polarity at local amplitude extrema
                # Polarity is positive at peaks and negative at troughs
                apparent_polarity = np.zeros_like(self.envelope_dict[self.file_name])
                apparent_polarity[maxima_indices] = np.cos(self.inst_phase_radians[self.file_name][maxima_indices])
                apparent_polarity[minima_indices] = -np.cos(self.inst_phase_radians[self.file_name][minima_indices])
                self.tensor_data = apparent_polarity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_App. Polarity", self.tensor_data)
                print('Apparent Polarity calculation done')
                QApplication.restoreOverrideCursor()
            elif self.envelope_dict.get(self.file_name) is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                self.envelope_dict[self.file_name] = np.abs(self.analytic_signal[self.file_name])
                # Find local amplitude maxima (peaks) and minima (troughs)
                # Find indices of local maxima along the time axis
                maxima_indices = argrelextrema(self.envelope_dict[self.file_name], np.greater, axis=0)
                # Find indices of local minima along the time axis
                minima_indices = argrelextrema(self.envelope_dict[self.file_name], np.less, axis=0)
                # Calculate apparent polarity at local amplitude extrema
                # Polarity is positive at peaks and negative at troughs
                apparent_polarity = np.zeros_like(self.envelope_dict[self.file_name])
                apparent_polarity[maxima_indices] = np.cos(self.inst_phase_radians[self.file_name][maxima_indices])
                apparent_polarity[minima_indices] = -np.cos(self.inst_phase_radians[self.file_name][minima_indices])
                self.tensor_data = apparent_polarity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_App. Polarity", self.tensor_data)
                print('Apparent Polarity calculation done')
                QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(self.custom_cursor)
                maxima_indices = argrelextrema(self.envelope_dict[self.file_name], np.greater, axis=0)
                # Find indices of local minima along the time axis
                minima_indices = argrelextrema(self.envelope_dict[self.file_name], np.less, axis=0)
                # Calculate apparent polarity at local amplitude extrema
                # Polarity is positive at peaks and negative at troughs
                apparent_polarity = np.zeros_like(self.envelope_dict[self.file_name])
                apparent_polarity[maxima_indices] = np.cos(self.inst_phase_radians[self.file_name][maxima_indices])
                apparent_polarity[minima_indices] = -np.cos(self.inst_phase_radians[self.file_name][minima_indices])
                self.tensor_data = apparent_polarity
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_App. Polarity", self.tensor_data)
                print('Apparent Polarity calculation done')
                QApplication.restoreOverrideCursor()

    def rms_amplitude(self):
        QApplication.setOverrideCursor(self.custom_cursor)
        # Pre-allocate the RMS amplitude tensor with the same shape as the input data
        window_size = 9
        rms_amplitude = np.zeros_like(self.tensor_data)

        # Calculate the number of samples along the time axis
        num_samples = self.tensor_data.shape[0]

        # Define the half window size
        half_window = window_size // 2

        # Loop over each time sample
        for i in range(half_window, num_samples - half_window):
            # Define the window range
            window = self.tensor_data[i - half_window:i + half_window + 1, :, :]

            # Square the values within the window
            squared_window = np.square(window)

            # Calculate the mean of the squared values
            mean_squared = np.mean(squared_window, axis=0)

            # Take the square root to get the RMS amplitude
            rms_amplitude[i, :, :] = np.sqrt(mean_squared)
        max_rms_amplitude = np.max(rms_amplitude)
        # Interpolate values at the edges
        # Create an interpolation function based on the valid RMS amplitude values
        x_valid = np.arange(half_window, num_samples - half_window)
        y_valid = rms_amplitude[half_window:num_samples - half_window, :, :]
        f_interp = interp1d(x_valid, y_valid, kind='linear', axis=0, fill_value='extrapolate')

        # Use the interpolation function to estimate the values at the edges
        x_edges = np.arange(0, half_window)
        y_edges_start = f_interp(x_edges)
        x_edges = np.arange(num_samples - half_window, num_samples)
        y_edges_end = f_interp(x_edges)

        # Fill in the estimated values at the start and end edges
        rms_amplitude[:half_window, :, :] = y_edges_start
        rms_amplitude[num_samples - half_window:, :, :] = y_edges_end

        # Check for negative frequencies and set them to zero if necessary
        rms_amplitude[rms_amplitude < 0] = 0

        # Check for negative frequencies and set them to zero if necessary
        rms_amplitude[rms_amplitude > max_rms_amplitude] = max_rms_amplitude

        self.tensor_data = rms_amplitude
        # Store the modified tensor with an informative key
        self.add_tensor(f"{self.file_name}_RMS Amplitude", self.tensor_data)
        print('RMS Amplitude calculation done')
        QApplication.restoreOverrideCursor()

    def process_seismic_data(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.segy_file_path, _ = QFileDialog.getOpenFileName(self, 'Open SEGY File', ''
                                                             , 'SEGY Files (*.segy *.sgy);;All Files (*)')

        QApplication.setOverrideCursor(self.custom_cursor)

        start_time = time.time()

        segy_file_name = os.path.splitext(os.path.basename(self.segy_file_path))[0]
        potential_columns = [
            'SourceX', 'SourceY', 'INLINE_3D', 'CROSSLINE_3D', 'TRACE_SEQUENCE_FILE',
            'CDP', 'EnergySourcePoint', 'FieldRecord', 'SourceEnergyDirectionExponent',
            'CDP_X', 'CDP_Y', 'ShotPoint', 'TraceValueMeasurementUnit', 'TransductionConstantMantissa'
        ]

        with segyio.open(self.segy_file_path, "r", ignore_geometry=True) as f:
            header_keys = segyio.tracefield.keys
            data = {k: f.attributes(header_keys[k])[:] for k in potential_columns if k in header_keys}
            trace_headers = pd.DataFrame(data)
            columns_to_keep = ['SourceX', 'SourceY', 'INLINE_3D', 'CROSSLINE_3D']

            # Renaming and adjusting columns based on conditions
            if (trace_headers[['SourceX', 'SourceY']] == 0).all().all():
                trace_headers = trace_headers.rename(
                    columns={'SourceX': 'tmp1', 'SourceY': 'tmp2', 'CDP_X': 'SourceX', 'CDP_Y': 'SourceY'})

            if (trace_headers[['INLINE_3D', 'CROSSLINE_3D']] == 0).all().all():
                trace_headers = trace_headers.rename(
                    columns={'INLINE_3D': 'tmp3', 'CROSSLINE_3D': 'tmp4', 'TRACE_SEQUENCE_FILE': 'INLINE_3D',
                             'CDP': 'CROSSLINE_3D'})
                if trace_headers['CROSSLINE_3D'].nunique() == len(trace_headers) and trace_headers[
                    'INLINE_3D'].nunique() > 1:
                    trace_headers = trace_headers.rename(
                        columns={'CROSSLINE_3D': 'tmp5', 'EnergySourcePoint': 'CROSSLINE_3D'})

            if (trace_headers['INLINE_3D'] == 0).all():
                trace_headers = trace_headers.rename(columns={'INLINE_3D': 'tmp6', 'FieldRecord': 'INLINE_3D'})

            if (trace_headers['SourceEnergyDirectionExponent'].nunique() > 1) and (
                    trace_headers['INLINE_3D'].nunique() * trace_headers['CROSSLINE_3D'].nunique() != len(
                trace_headers)):
                trace_headers = trace_headers.rename(
                    columns={'INLINE_3D': 'tmp7', 'SourceEnergyDirectionExponent': 'INLINE_3D'})

            if (trace_headers['INLINE_3D'] == 0).all():
                trace_headers = trace_headers.rename(
                    columns={'INLINE_3D': 'tmp8', 'CROSSLINE_3D': 'tmp9', 'CDP_X': 'INLINE_3D',
                             'CDP_Y': 'CROSSLINE_3D'})

            if ('CDP_X' in trace_headers.columns) and ('CDP_Y' in trace_headers.columns) and (
                    trace_headers[['INLINE_3D', 'CROSSLINE_3D']].max() > 10000).all():
                trace_headers = trace_headers.rename(
                    columns={'INLINE_3D': 'tmp10', 'CROSSLINE_3D': 'tmp11', 'CDP_X': 'INLINE_3D',
                             'CDP_Y': 'CROSSLINE_3D'})

            if trace_headers['INLINE_3D'].max() > 15000:
                trace_headers = trace_headers.rename(
                    columns={'INLINE_3D': 'tmp12', 'CROSSLINE_3D': 'tmp13', 'ShotPoint': 'INLINE_3D',
                             'TraceValueMeasurementUnit': 'CROSSLINE_3D'})

            if (trace_headers['CROSSLINE_3D'] == 0).all():
                if trace_headers['SourceY'].max() < 10000:
                    trace_headers = trace_headers.rename(
                        columns={'CROSSLINE_3D': 'tmp14', 'SourceY': 'CROSSLINE_3D', 'SourceX': 'tmp15',
                                 'TraceValueMeasurementUnit': 'SourceX', 'TransductionConstantMantissa': 'SourceY'})
                else:
                    trace_headers = trace_headers.rename(columns={'CROSSLINE_3D': 'tmp16', 'CDP_Y': 'CROSSLINE_3D'})

            final_headers = trace_headers[columns_to_keep].copy()

            # Ensure unique values for INLINE_3D and CROSSLINE_3D
            unique_inline3d_values = np.array(sorted(final_headers['INLINE_3D'].unique()))
            unique_crossline3d_values = np.array(sorted(final_headers['CROSSLINE_3D'].unique()))

            inline3d_index = {value: index for index, value in enumerate(unique_inline3d_values)}
            crossline3d_index = {value: index for index, value in enumerate(unique_crossline3d_values)}

            # Calculate dimensions of the 3D array
            inline3d_max = len(unique_inline3d_values)
            crossline3d_max = len(unique_crossline3d_values)

            max_sourcex = final_headers['SourceX'].max()
            min_sourcex = final_headers[final_headers['SourceX'] != 0]['SourceX'].min()
            max_sourcey = final_headers['SourceY'].max()
            min_sourcey = final_headers[final_headers['SourceY'] != 0]['SourceY'].min()

            # Print results
            print(f"Unique INLINE_3D values: {inline3d_max}")
            print(f"Unique CROSSLINE_3D values: {crossline3d_max}")
            print(f"Max SourceX: {max_sourcex}")
            print(f"Min SourceX: {min_sourcex}")
            print(f"Max SourceY: {max_sourcey}")
            print(f"Min SourceY: {min_sourcey}")

            # Extract amplitude data
            amplitudes = [f.trace[i] for i in range(f.tracecount)]
            num_time_intervals = len(amplitudes[0])

            # Access the Sample Interval (in ms) from the Binary Header
            sample_interval = f.bin[segyio.BinField.Interval] / 1000
            print(f"Sampling Interval: {sample_interval} ms")

            # Initialize 3D array with zeros, setting dtype to float32
            seismic_cube = np.zeros((inline3d_max, crossline3d_max, num_time_intervals), dtype=np.float32)

            # Populate the 3D array based on unique values
            for idx, row in final_headers.iterrows():
                inline_idx = inline3d_index[row['INLINE_3D']]
                crossline_idx = crossline3d_index[row['CROSSLINE_3D']]
                seismic_cube[inline_idx, crossline_idx, :] = amplitudes[idx]

            seismic_cube = np.expand_dims(seismic_cube, axis=-1)  # Adding channel dimension

            # Transpose the tensor to match the desired orientation
            transposed_tensor = np.transpose(seismic_cube, (2, 0, 1, 3))

            # Save the final tensor as a .npy file with min/max values in the filename
            np.save(
                f"{segy_file_name}_tensor_{min_sourcex}_{max_sourcex}_{min_sourcey}_{max_sourcey}_sampling_interval{sample_interval}ms.npy",
                transposed_tensor)

            # Print the number of dimensions and shape of the final tensor
            print("Final Tensor Number of Dimensions:", transposed_tensor.ndim)
            print("Final Tensor Shape:", transposed_tensor.shape)

            # Record the end time
            end_time = time.time()

            # Calculate and print the execution time
            execution_time = end_time - start_time
            print("Execution Time:", execution_time, "seconds")

            # Replace the base name (filename) with a new value
            new_base_name = f"{segy_file_name}_tensor_{min_sourcex}_{max_sourcex}_{min_sourcey}_{max_sourcey}_sampling_interval{sample_interval}ms.npy"

            # Extract the base name (filename) from the path
            base_name = os.path.basename(new_base_name)

            # Split the base name by underscore '_'
            parts = base_name.split('_')

            self.file_name = parts[0]

            print("New file path:", new_base_name)
            self.file_path = new_base_name
            self.loaded_file_paths[self.file_name] = new_base_name
            self.update_recent_files_submenu()  # Update the recent files submenu
            self.tensor_data = transposed_tensor
            self.add_tensor(self.file_name, self.tensor_data)
            print(f"Successfully loaded tensor. Shape: {self.tensor_data.shape}")
            QApplication.restoreOverrideCursor()

    def tensor_cutter(self):
        # Create a QDialog instance
        dialog = QDialog(self)
        dialog.setWindowTitle('Slice 4D Tensor')

        layout = QVBoxLayout(dialog)

        # Depth slice (1st dimension)
        self.depth_min, self.depth_max = self.create_slider_and_spinbox('Depth', layout, self.tensor_data.shape[0])
        # Height slice (2nd dimension)
        self.height_min, self.height_max = self.create_slider_and_spinbox('Height', layout, self.tensor_data.shape[1])
        # Width slice (3rd dimension)
        self.width_min, self.width_max = self.create_slider_and_spinbox('Width', layout, self.tensor_data.shape[2])

        # Cut Button
        cut_button = QPushButton('Cut', dialog)
        cut_button.clicked.connect(lambda: self.slice_tensor(dialog))
        layout.addWidget(cut_button)

        dialog.setLayout(layout)
        dialog.exec_()  # Execute the dialog

    def create_slider_and_spinbox(self, label, layout, max_value):
        """ Helper method to create a label, a slider and a spinbox for each dimension. """
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))

        min_spinbox = QSpinBox()
        min_spinbox.setRange(1, max_value)
        min_spinbox.setValue(1)

        max_spinbox = QSpinBox()
        max_spinbox.setRange(1, max_value)
        max_spinbox.setValue(max_value)

        min_slider = QSlider()
        min_slider.setRange(1, max_value)
        min_slider.setValue(1)
        min_slider.setOrientation(1)  # Horizontal

        max_slider = QSlider()
        max_slider.setRange(1, max_value)
        max_slider.setValue(max_value)
        max_slider.setOrientation(1)  # Horizontal

        # Connect spinboxes and sliders
        min_spinbox.valueChanged.connect(min_slider.setValue)
        min_slider.valueChanged.connect(min_spinbox.setValue)
        max_spinbox.valueChanged.connect(max_slider.setValue)
        max_slider.valueChanged.connect(max_spinbox.setValue)

        hbox.addWidget(min_slider)
        hbox.addWidget(min_spinbox)
        hbox.addWidget(QLabel('to'))
        hbox.addWidget(max_spinbox)
        hbox.addWidget(max_slider)

        layout.addLayout(hbox)

        return min_spinbox, max_spinbox

    def slice_tensor(self, dialog):
        dialog.accept()
        # Set custom cursor (already part of your application context)
        QApplication.setOverrideCursor(self.custom_cursor)
        """ Perform the slicing of the tensor based on the selected ranges. """
        # Convert 1-based indices to 0-based indices for slicing
        d_min = self.depth_min.value() - 1
        d_max = self.depth_max.value()  # inclusive in slicing terms
        h_min = self.height_min.value() - 1
        h_max = self.height_max.value()
        w_min = self.width_min.value() - 1
        w_max = self.width_max.value()

        # Assuming the tensor to slice is passed or accessible within the class
        # This is a placeholder tensor for the purpose of the example
        tensor = np.copy(self.tensor_data)

        # Perform the slicing
        output = tensor[d_min:d_max, h_min:h_max, w_min:w_max, :]

        # Extract the file name from the file path
        filename = os.path.basename(self.file_path)

        # Extract the min and max values from the filename
        parts = filename.split('_tensor_')[-1].split('_')
        x_min, x_max, y_min, y_max = map(float,
                                         parts[:4])  # Only take the first four values after 'tensor_'

        print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

        def calculate_new_global_coordinates(
                original_shape,
                global_min_x, global_max_x, global_min_y, global_max_y,
                local_min_x, local_max_x, local_min_y, local_max_y):

            # Extract the dimensions
            original_height, original_width = original_shape

            # Calculate the resolution (distance per pixel in global coordinates)
            x_resolution = (global_max_x - global_min_x) / original_width
            y_resolution = (global_max_y - global_min_y) / original_height

            # Convert local cut coordinates to global coordinates
            new_global_min_x = global_min_x + local_min_x * x_resolution
            new_global_max_x = global_min_x + local_max_x * x_resolution

            new_global_min_y = global_min_y + local_min_y * y_resolution
            new_global_max_y = global_min_y + local_max_y * y_resolution

            return new_global_min_x, new_global_max_x, new_global_min_y, new_global_max_y

        height = self.tensor_data.shape[1]
        width = self.tensor_data.shape[2]
        original_shape = (height, width)
        global_min_x, global_max_x = x_min, x_max
        global_min_y, global_max_y = y_min, y_max

        # Local cut coordinates
        local_min_x, local_max_x = self.height_min.value(), self.height_max.value()
        local_min_y, local_max_y = self.width_min.value(), self.width_max.value()

        # Calculate new global coordinates
        min_sourcex, max_sourcex, min_sourcey, max_sourcey = calculate_new_global_coordinates(
            original_shape,
            global_min_x, global_max_x, global_min_y, global_max_y,
            local_min_x, local_max_x, local_min_y, local_max_y
        )

        new_coords = (min_sourcex, max_sourcex, min_sourcey, max_sourcey)

        print("New global coordinates after cut:", new_coords)

        sampling_interval = self.get_sampling_interval_from_file_name()

        # Save the final tensor as a .npy file with min/max values in the filename
        np.save(
            f"{self.file_name}{output.shape}_tensor_{min_sourcex}_{max_sourcex}_{min_sourcey}_{max_sourcey}_sampling_interval{sampling_interval}ms.npy",
            output)

        # Replace the base name (filename) with a new value
        new_base_name = f"{self.file_name}{output.shape}_tensor_{min_sourcex}_{max_sourcex}_{min_sourcey}_{max_sourcey}_sampling_interval{sampling_interval}ms.npy"

        # Extract the base name (filename) from the path
        base_name = os.path.basename(new_base_name)

        # Split the base name by underscore '_'
        parts = base_name.split('_')

        self.file_name = parts[0]

        print("New file path:", new_base_name)
        self.file_path = new_base_name
        self.loaded_file_paths[self.file_name] = new_base_name
        self.update_recent_files_submenu()  # Update the recent files submenu
        self.tensor_data = output
        self.add_tensor(self.file_name, self.tensor_data)
        print(f"Successfully loaded tensor. Shape: {self.tensor_data.shape}")
        QApplication.restoreOverrideCursor()

    def fault_prediction(self):
        batch_size = 6
        depth = 128
        height = 128
        width = 128
        completion_event = multiprocessing.Event()
        # Create queues for communication
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        # Prepare the input data (e.g., input_data)
        input_data = self.tensor_data  # The data you need to pass to the process
        # Put the input data into the input queue
        input_queue.put(input_data)

        # Function to open a dialog and return the selected model path
        def get_model_path():
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setNameFilter("HDF5 Files (*.h5 *.pth)")
            if dialog.exec_():
                model_path = dialog.selectedFiles()
                return model_path[0]  # Return the first selected file
            return None  # Return None if no file is selected

        # Use the function to get the model path
        model_path = get_model_path()
        if model_path:
            # Check the file extension and load the model accordingly
            if model_path.endswith('.h5'):
                # Define the styles for light and dark themes
                light_theme_style = """
                QDialog {
                    background-color: #B2B2B2;
                    color: #333333;
                }
                """
                dark_theme_style = """
                QWidget {
                    background-color: #333333;
                    color: #FFFFFF;
                }
                """

                # Function to create a styled QInputDialog for integer input
                def create_styled_input_dialog(title, label, initial_value):
                    dialog = QInputDialog()

                    if self.isDarkTheme:
                        dialog.setStyleSheet(dark_theme_style)
                    else:
                        dialog.setStyleSheet(light_theme_style)

                    dialog.setWindowTitle(title)
                    dialog.setLabelText(label)
                    dialog.setIntRange(1, 100)  # Set the range of acceptable values
                    dialog.setIntValue(initial_value)  # Set the initial value
                    dialog.exec_()  # Execute the dialog
                    return dialog.intValue()  # Return the value entered by the user

                # Usage example
                batch_size = create_styled_input_dialog("Input", "Define batch size", 6)

            elif model_path.endswith('.pth'):
                # Define the styles for light and dark themes
                light_theme_style = """
                QDialog {
                    background-color: #B2B2B2;  /* Light grey background */
                    color: #333333;  /* Dark grey text */
                }
                QWidget {
                    background-color: #B2B2B2;  /* Light grey background */
                    color: #333333;  /* Dark grey text */
                }
                """

                dark_theme_style = """
                QDialog {
                    background-color: #333333;  /* Dark grey background */
                    color: #FFFFFF;  /* White text for better readability */
                }
                QWidget {
                    background-color: #333333;  /* Dark grey background */
                    color: #FFFFFF;  /* White text for better readability */
                }
                """

                # Function to create a styled QInputDialog for integer input
                def create_styled_input_dialog(title, label, initial_value):
                    dialog = QInputDialog()

                    # Apply the selected theme style
                    if self.isDarkTheme:
                        dialog.setStyleSheet(dark_theme_style)
                    else:
                        dialog.setStyleSheet(light_theme_style)

                    dialog.setWindowTitle(title)
                    dialog.setLabelText(label)
                    dialog.setIntRange(1, 10000)  # Adjust range as per your application's needs
                    dialog.setIntValue(initial_value)

                    ok = dialog.exec_()  # Execute the dialog
                    return dialog.intValue(), ok

                # Step 2: Get the depth, height, and width from the user using styled input dialogs
                depth, ok1 = create_styled_input_dialog("Input Depth", "Enter the depth of the input data:", 128)
                if not ok1:
                    raise ValueError("User cancelled the input dialog for depth.")

                height, ok2 = create_styled_input_dialog("Input Height", "Enter the height of the input data:", 128)
                if not ok2:
                    raise ValueError("User cancelled the input dialog for height.")

                width, ok3 = create_styled_input_dialog("Input Width", "Enter the width of the input data:", 128)
                if not ok3:
                    raise ValueError("User cancelled the input dialog for width.")

        # Set custom cursor (already part of your application context)
        QApplication.setOverrideCursor(self.custom_cursor)

        # Create a new process for the inference task, passing 'self.isDarkTheme' as well
        process = multiprocessing.Process(target=TensorVisualizer.run_inference, args=(input_queue, output_queue,
                                                                                       model_path, batch_size, depth,
                                                                                       height, width, completion_event))
        # Start the process
        process.start()

        # Wait for the process to signal completion
        while not completion_event.is_set():
            process.join(timeout=1)  # Check every second if the event is set

        # Wait for the process to finish
        result = output_queue.get()

        # Terminate the process if it's still alive
        if process.is_alive():
            print("Process finished, terminating...")
            process.terminate()
            process.join()  # Ensure the process has been terminated

        self.tensor_data = result
        # Store the modified tensor with an informative key
        self.add_tensor(f"{self.file_name}_Faults", self.tensor_data)
        print('Fault Prediction done')

        QApplication.restoreOverrideCursor()

    @staticmethod
    def run_inference(input_queue, output_queue, model_path, batch_size, depth, height, width, completion_event):
        try:
            data = input_queue.get()

            if model_path:
                # Check the file extension and load the model accordingly
                if model_path.endswith('.h5'):

                    model = load_model(model_path, compile=False)
                    # Retrieve the expected input shape from the model
                    input_shape = model.input_shape

                    # Exclude the batch size dimension and retrieve the target dimensions
                    if len(input_shape) == 5:  # For 3D CNNs
                        _, target_depth, target_height, target_width, _ = input_shape

                        # Ensure the target dimensions are initialized properly
                        if target_depth is None or target_height is None or target_width is None:
                            raise ValueError(
                                "Model input shape dimensions not properly set for target depth, height, and width.")

                        # Normalization (Assuming the data is 4D: (depth, height, width, channel))
                        X_train = data
                        mean_X = np.mean(X_train)
                        std_X = np.std(X_train)
                        X_train_normalized = (X_train - mean_X) / std_X

                        # Print the shape of the normalized input tensor
                        print("Shape of Normalized Input Tensor:", X_train_normalized.shape)

                        # Add batch dimension and adjust to model's input shape
                        X_train_expanded = np.expand_dims(X_train_normalized, axis=0)  # Add batch dimension

                        # Resize to match target depth, height, width, and channel
                        # Assuming resize function handles 5D resizing: (batch_size, depth, height, width, channels)
                        X_train_resized = resize(X_train_expanded,
                                                 (1, target_depth, target_height, target_width, 1))

                        # Print the shape of the resized input tensor
                        print("Shape of Resized Input Tensor:", X_train_resized.shape)

                        # Predict using the trained model
                        predicted_output = model.predict(X_train_resized, verbose=0)

                        # Print the shape of the predicted output tensor
                        print("Shape of Predicted Output Tensor:", predicted_output.shape)

                        # Remove batch dimension for the final output
                        final_predicted_output = np.squeeze(predicted_output, axis=0)  # Remove batch

                        # Apply thresholding to the predicted output
                        final_predicted_output[final_predicted_output < 0.2] = 0
                        final_predicted_output[final_predicted_output >= 0.2] = 1

                        # Send the result back via the output queue
                        output_queue.put(final_predicted_output)

                        # Signal completion to the parent process
                        completion_event.set()

                    elif len(input_shape) == 4:  # For 2D CNNs
                        _, target_height, target_width, _ = input_shape

                        # Ensure target_height and target_width are initialized properly
                        if target_height is None or target_width is None:
                            raise ValueError(
                                "Model input shape dimensions not properly set for target height and width.")

                        # Transpose the data to feed the model from the side
                        X_train_transposed = np.transpose(data, (2, 0, 1, 3))  # Inline (1, 0, 2, 3)

                        # Z-score normalize the data
                        mean_X = np.mean(X_train_transposed)
                        std_X = np.std(X_train_transposed)
                        X_train_normalized = (X_train_transposed - mean_X) / std_X

                        # Print the shape of the normalized input tensor
                        print("Shape of Normalized Input Tensor:", X_train_normalized.shape)

                        # Predict using the trained model in batches
                        predicted_output_batches = []
                        num_samples = X_train_normalized.shape[0]
                        for i in range(0, num_samples, batch_size):
                            batch_data = X_train_normalized[i:i + batch_size]

                            # Resize each image in the batch to the model's input size
                            resized_batch_data = np.zeros(
                                (batch_data.shape[0], target_height, target_width, batch_data.shape[3]))
                            for j in range(batch_data.shape[0]):
                                resized_batch_data[j] = resize(batch_data[j], (target_height, target_width))

                            # Predict on resized batch
                            batch_output = model.predict(resized_batch_data, verbose=0)

                            # Resize the predicted output back to the original image size
                            resized_batch_output = np.zeros((batch_output.shape[0], X_train_transposed.shape[1],
                                                             X_train_transposed.shape[2], batch_output.shape[3]))
                            for j in range(batch_output.shape[0]):
                                resized_batch_output[j] = resize(batch_output[j],
                                                                 (X_train_transposed.shape[1],
                                                                  X_train_transposed.shape[2]))

                            predicted_output_batches.append(resized_batch_output)

                        # Concatenate predicted output batches
                        predicted_output = np.concatenate(predicted_output_batches, axis=0)

                        # Print the shape of the predicted output tensor
                        print("Shape of Predicted Output Tensor:", predicted_output.shape)

                        final_predicted_output = np.transpose(predicted_output, (1, 2, 0, 3))

                        final_predicted_output[final_predicted_output < 0.2] = 0

                        final_predicted_output[final_predicted_output >= 0.2] = 1

                        # Send the result back via the output queue
                        output_queue.put(final_predicted_output)

                        # Signal completion to the parent process
                        completion_event.set()
                    else:
                        raise ValueError("Unsupported input shape dimensions. Expected 4D or 5D input shape.")

                elif model_path.endswith('.pth'):

                    model_input_size = (depth, height, width)

                    # Step 3: Load the model
                    model = torch.load(model_path)
                    model.eval()  # Set the model to evaluation mode

                    # Move the model to the appropriate device (GPU if available, otherwise CPU)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)

                    # Load and normalize the input data
                    input_data = np.copy(data)
                    mean_input = np.mean(input_data)
                    std_input = np.std(input_data)
                    normalized_input = (input_data - mean_input) / std_input

                    # Assume input data is in the shape (depth, height, width, channel)
                    depth_size, height_size, width_size, channels = normalized_input.shape

                    # Function to generate sliding windows
                    def generate_windows(data, window_size, stride):
                        depth, height, width, _ = data.shape
                        for d in range(0, depth - window_size[0] + 1, stride[0]):
                            for h in range(0, height - window_size[1] + 1, stride[1]):
                                for w in range(0, width - window_size[2] + 1, stride[2]):
                                    yield d, h, w, data[d:d + window_size[0], h:h + window_size[1],
                                                   w:w + window_size[2], :]
                        # Ensure last window covers the end of the array
                        if depth % window_size[0] != 0:
                            for h in range(0, height - window_size[1] + 1, stride[1]):
                                for w in range(0, width - window_size[2] + 1, stride[2]):
                                    yield depth - window_size[0], h, w, data[depth - window_size[0]:,
                                                                        h:h + window_size[1],
                                                                        w:w + window_size[2], :]
                        if height % window_size[1] != 0:
                            for d in range(0, depth - window_size[0] + 1, stride[0]):
                                for w in range(0, width - window_size[2] + 1, stride[2]):
                                    yield d, height - window_size[1], w, data[d:d + window_size[0],
                                                                         height - window_size[1]:,
                                                                         w:w + window_size[2], :]
                        if width % window_size[2] != 0:
                            for d in range(0, depth - window_size[0] + 1, stride[0]):
                                for h in range(0, height - window_size[1] + 1, stride[1]):
                                    yield d, h, width - window_size[2], data[d:d + window_size[0], h:h + window_size[1],
                                                                        width - window_size[2]:, :]
                        # Cover the corners
                        if depth % window_size[0] != 0 and height % window_size[1] != 0:
                            for w in range(0, width - window_size[2] + 1, stride[2]):
                                yield depth - window_size[0], height - window_size[1], w, data[depth - window_size[0]:,
                                                                                          height - window_size[1]:,
                                                                                          w:w + window_size[2], :]
                        if depth % window_size[0] != 0 and width % window_size[2] != 0:
                            for h in range(0, height - window_size[1] + 1, stride[1]):
                                yield depth - window_size[0], h, width - window_size[2], data[depth - window_size[0]:,
                                                                                         h:h + window_size[1],
                                                                                         width - window_size[2]:, :]
                        if height % window_size[1] != 0 and width % window_size[2] != 0:
                            for d in range(0, depth - window_size[0] + 1, stride[0]):
                                yield d, height - window_size[1], width - window_size[2], data[d:d + window_size[0],
                                                                                          height - window_size[1]:,
                                                                                          width - window_size[2]:, :]
                        if depth % window_size[0] != 0 and height % window_size[1] != 0 and width % window_size[2] != 0:
                            yield depth - window_size[0], height - window_size[1], width - window_size[2], data[
                                                                                                           depth -
                                                                                                           window_size[
                                                                                                               0]:,
                                                                                                           height -
                                                                                                           window_size[
                                                                                                               1]:,
                                                                                                           width -
                                                                                                           window_size[
                                                                                                               2]:,
                                                                                                           :]

                    # Initialize an array to accumulate the output
                    output_data = np.zeros((depth_size, height_size, width_size, channels))

                    # Weights for averaging overlapping regions
                    overlap_count = np.zeros((depth_size, height_size, width_size, channels))

                    stride = (
                        model_input_size[0] // 2, model_input_size[1] // 2, model_input_size[2] // 2)  # Overlap by 50%

                    for d, h, w, window in generate_windows(normalized_input, model_input_size, stride):
                        # Prepare the window for the model
                        window = np.expand_dims(window, axis=0)  # Add batch dimension
                        window_tensor = torch.from_numpy(window).permute(0, 4, 1, 2, 3).float().to(
                            device)  # Shape: (batch, channel, depth, height, width)

                        # Perform inference
                        with torch.no_grad():
                            output = model(window_tensor)

                        # Convert the output back to numpy and remove the batch dimension
                        output = output.squeeze(0).permute(1, 2, 3,
                                                           0).cpu().numpy()  # Shape: (depth, height, width, channel)

                        # Accumulate the output in the appropriate location in the output array
                        output_data[d:d + model_input_size[0], h:h + model_input_size[1], w:w + model_input_size[2],
                        :] += output
                        overlap_count[d:d + model_input_size[0], h:h + model_input_size[1], w:w + model_input_size[2],
                        :] += 1

                    # Normalize the accumulated results by the overlap count
                    output_data /= np.maximum(overlap_count, 1)

                    # Apply thresholding to the output
                    output_data[output_data < 0.9] = 0
                    output_data[output_data >= 0.9] = 1

                    # Send the result back via the output queue
                    output_queue.put(output_data)

                    # Signal completion to the parent process
                    completion_event.set()

            else:
                print("No model selected.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical("Error", f"An error occurred: {e}")

    def facies_prediction(self):
        try:
            # Function to open a dialog and return the selected model path
            def get_model_path():
                dialog = QFileDialog()
                dialog.setFileMode(QFileDialog.ExistingFile)
                dialog.setNameFilter("HDF5 Files (*.h5)")
                if dialog.exec_():
                    QApplication.setOverrideCursor(self.custom_cursor)
                    model_path = dialog.selectedFiles()
                    return model_path[0]  # Return the first selected file
                return None  # Return None if no file is selected

            # Use the function to get the model path
            model_path = get_model_path()
            if model_path:
                model = load_model(model_path, compile=False)
                # Retrieve the expected input shape from the model
                input_shape = model.input_shape

                # Exclude the batch size dimension and retrieve the target dimensions
                if len(input_shape) == 5:  # For 3D CNNs
                    _, target_depth, target_height, target_width, _ = input_shape

                    # Ensure the target dimensions are initialized properly
                    if target_depth is None or target_height is None or target_width is None:
                        raise ValueError(
                            "Model input shape dimensions not properly set for target depth, height, and width.")

                    # Normalization (Assuming the data is 4D: (depth, height, width, channel))
                    X_train = self.tensor_data
                    mean_X = np.mean(X_train)
                    std_X = np.std(X_train)
                    X_train_normalized = (X_train - mean_X) / std_X

                    # Print the shape of the normalized input tensor
                    print("Shape of Normalized Input Tensor:", X_train_normalized.shape)

                    # Add batch dimension and adjust to model's input shape
                    X_train_expanded = np.expand_dims(X_train_normalized, axis=0)  # Add batch dimension

                    # Resize to match target depth, height, width, and channel
                    # Assuming resize function handles 5D resizing: (batch_size, depth, height, width, channels)
                    X_train_resized = resize(X_train_expanded,
                                             (1, target_depth, target_height, target_width, 1))

                    # Print the shape of the resized input tensor
                    print("Shape of Resized Input Tensor:", X_train_resized.shape)

                    # Predict using the trained model
                    predicted_output = model.predict(X_train_resized, verbose=0)

                    # Print the shape of the predicted output tensor
                    print("Shape of Predicted Output Tensor:", predicted_output.shape)

                    # Remove batch dimension for the final output
                    final_predicted_output = np.squeeze(predicted_output, axis=0)  # Remove batch

                    self.tensor_data = final_predicted_output
                    # Store the modified tensor with an informative key
                    self.add_tensor(f"{self.file_name}_Facies", self.tensor_data)
                    print('Facies Prediction done')
                    QApplication.restoreOverrideCursor()

                elif len(input_shape) == 4:  # For 2D CNNs
                    _, target_height, target_width, _ = input_shape

                    # Ensure target_height and target_width are initialized properly
                    if target_height is None or target_width is None:
                        raise ValueError("Model input shape dimensions not properly set for target height and width.")

                    # Transpose the data to feed the model from the side
                    X_train_transposed = np.transpose(self.tensor_data, (2, 0, 1, 3))  # Inline (1, 0, 2, 3)

                    # Z-score normalize the data
                    mean_X = np.mean(X_train_transposed)
                    std_X = np.std(X_train_transposed)
                    X_train_normalized = (X_train_transposed - mean_X) / std_X

                    # Print the shape of the normalized input tensor
                    print("Shape of Normalized Input Tensor:", X_train_normalized.shape)

                    # Define the styles for light and dark themes
                    light_theme_style = """
                    QDialog {
                        background-color: #B2B2B2;
                        color: #333333;
                    }
                    """
                    dark_theme_style = """
                    QWidget {
                        background-color: #333333;
                        color: #FFFFFF;
                    }
                    """

                    # Function to create a styled QInputDialog for integer input
                    def create_styled_input_dialog(title, label, initial_value):
                        QApplication.restoreOverrideCursor()
                        dialog = QInputDialog()

                        if self.isDarkTheme:
                            dialog.setStyleSheet(dark_theme_style)
                        else:
                            dialog.setStyleSheet(light_theme_style)

                        dialog.setWindowTitle(title)
                        dialog.setLabelText(label)
                        dialog.setIntRange(1, 100)  # Set the range of acceptable values
                        dialog.setIntValue(initial_value)  # Set the initial value
                        dialog.exec_()  # Execute the dialog
                        QApplication.setOverrideCursor(self.custom_cursor)
                        return dialog.intValue()  # Return the value entered by the user

                    # Usage example
                    batch_size = create_styled_input_dialog("Input", "Define batch size", 6)

                    # Predict using the trained model in batches
                    predicted_output_batches = []
                    num_samples = X_train_normalized.shape[0]
                    for i in range(0, num_samples, batch_size):
                        batch_data = X_train_normalized[i:i + batch_size]

                        # Resize each image in the batch to the model's input size
                        resized_batch_data = np.zeros(
                            (batch_data.shape[0], target_height, target_width, batch_data.shape[3]))
                        for j in range(batch_data.shape[0]):
                            resized_batch_data[j] = resize(batch_data[j], (target_height, target_width))

                        # Predict on resized batch
                        batch_output = model.predict(resized_batch_data, verbose=0)

                        # Resize the predicted output back to the original image size
                        resized_batch_output = np.zeros((batch_output.shape[0], X_train_transposed.shape[1],
                                                         X_train_transposed.shape[2], batch_output.shape[3]))
                        for j in range(batch_output.shape[0]):
                            resized_batch_output[j] = resize(batch_output[j],
                                                             (X_train_transposed.shape[1], X_train_transposed.shape[2]))

                        predicted_output_batches.append(resized_batch_output)

                    # Concatenate predicted output batches
                    predicted_output = np.concatenate(predicted_output_batches, axis=0)

                    # Print the shape of the predicted output tensor
                    print("Shape of Predicted Output Tensor:", predicted_output.shape)

                    final_predicted_output = np.transpose(predicted_output, (1, 2, 0, 3))

                    self.tensor_data = final_predicted_output
                    # Store the modified tensor with an informative key
                    self.add_tensor(f"{self.file_name}_Facies", self.tensor_data)
                    print('Facies Prediction done')
                    QApplication.restoreOverrideCursor()
                else:
                    QApplication.restoreOverrideCursor()
                    raise ValueError("Unsupported input shape dimensions. Expected 4D or 5D input shape.")
            else:
                QApplication.restoreOverrideCursor()
                print("No model selected.")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def thresholded_amplitude(self):
        # Step 1: Flatten the 'channel' dimension to compute the mean and standard deviation
        Data = np.copy(self.tensor_data)
        amplitude_values = Data[:, :, :, -1].flatten()  # Assuming amplitude values are in the last channel

        # Step 2: Calculate the mean and standard deviation for z-score normalization
        mean_amplitude = np.mean(amplitude_values)
        std_amplitude = np.std(amplitude_values)

        # Step 3: Normalize the amplitude values using z-score
        normalized_data = (Data[:, :, :, -1] - mean_amplitude) / std_amplitude
        normalized_data_min = np.min(normalized_data)
        normalized_data_max = np.max(normalized_data)
        print(normalized_data_min)
        print(normalized_data_max)

        # Define the styles for light and dark themes
        light_theme_style = """
        QDialog {
            background-color: #B2B2B2; /* Light grey background */
            color: #333333; /* Dark grey text */
        }
        /* Add other widget styles for light theme here */
        """
        dark_theme_style = """
        QWidget {
            background-color: #333333; /* Dark grey background */
            color: #FFFFFF; /* White text for better readability */
        }
        /* Add other widget styles for dark theme here */
        """

        # Function to create a styled QInputDialog
        def create_styled_input_dialog(title, label, initial_value):
            dialog = QInputDialog()
            if self.isDarkTheme:
                dialog.setStyleSheet(dark_theme_style)
            else:
                dialog.setStyleSheet(light_theme_style)
            dialog.setWindowTitle(title)
            dialog.setLabelText(label)
            dialog.setDoubleDecimals(2)
            dialog.setDoubleMinimum(-float('inf'))
            dialog.setDoubleMaximum(float('inf'))
            dialog.setDoubleValue(initial_value)
            return dialog

        # Step 4: Get user input for threshold values using styled QInputDialog
        ok = False
        while not ok:
            dialog = create_styled_input_dialog('Input Threshold', 'Enter minimum threshold value:', -2.00)
            ok = dialog.exec_() == QDialog.Accepted
            if not ok:
                # Handle the case where the user cancels the input dialog
                return
            threshold_min = dialog.doubleValue()

        ok = False
        while not ok:
            dialog = create_styled_input_dialog('Input Threshold', 'Enter maximum threshold value:', 2.00)
            ok = dialog.exec_() == QDialog.Accepted
            if not ok:
                # Handle the case where the user cancels the input dialog
                return
            threshold_max = dialog.doubleValue()

        QApplication.setOverrideCursor(self.custom_cursor)

        normalized_data[(normalized_data >= threshold_min) & (normalized_data <= threshold_max)] = 0

        # Step 5: Replace the original amplitude values with the thresholded values
        Data[:, :, :, -1] = normalized_data

        self.tensor_data = Data
        # Store the modified tensor with an informative key
        self.add_tensor(f"{self.file_name}_Thresholded Amplitude", self.tensor_data)
        print('Thresholded Amplitude calculation done')

        QApplication.restoreOverrideCursor()

    def kmeans_amplitude(self):
        self.num_clusters[self.file_name], okPressed = QInputDialog.getInt(
            self, "Input Number of Clusters", "Number of Clusters:", 10, 1, 100, 1)
        if okPressed:
            if self.kmeans_dict.get(f"{self.file_name}_{self.num_clusters[self.file_name]}") is None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Flatten the 'channel' dimension to compute the mean and standard deviation
                self.tensor_data = np.copy(self.tensor_data)
                amplitude_values = self.tensor_data[:, :, :, -1].flatten()

                # Calculate the mean and standard deviation for z-score normalization
                mean_amplitude = np.mean(amplitude_values)
                std_amplitude = np.std(amplitude_values)

                # Normalize the amplitude values using z-score
                normalized_data = (self.tensor_data[:, :, :, -1] - mean_amplitude) / std_amplitude

                # Reshape normalized data for K-Means
                reshaped_data = normalized_data.reshape(-1, 1)

                # Apply K-Means clustering
                kmeans = KMeans(n_clusters=self.num_clusters[self.file_name], random_state=42)
                kmeans.fit(reshaped_data)

                # Get the cluster labels for each data point
                cluster_labels = kmeans.labels_

                # Replace the original amplitude values with the cluster labels
                self.tensor_data[:, :, :, -1] = cluster_labels.reshape(self.tensor_data.shape[:-1])

                # Ensure class 0 is not used by incrementing all cluster labels by 1
                self.tensor_data[:, :, :, -1] += 1
                uni = np.unique(self.tensor_data)
                print(uni)

                copied_original_data = np.copy(self.tensor_data)
                self.kmeans_dict[f"{self.file_name}_{self.num_clusters[self.file_name]}"] = copied_original_data

                QApplication.restoreOverrideCursor()

                # After clustering, prompt the user to select which clusters to keep
                clusters_to_keep = self.select_clusters_to_keep(self.num_clusters[self.file_name])
                QApplication.setOverrideCursor(self.custom_cursor)
                if clusters_to_keep is not None:
                    print('keep')
                    # Set the values of the clusters not selected to zero
                    for i in range(1, self.num_clusters[self.file_name] + 1):
                        if i not in clusters_to_keep:
                            self.tensor_data[self.tensor_data[:, :, :, -1] == i] = 0

                uni3 = np.unique(self.tensor_data)
                print(uni3)
                uni2 = np.unique(self.kmeans_dict[f"{self.file_name}_{self.num_clusters[self.file_name]}"])
                print('original', uni2)
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_KMeans Clustered Amplitude {self.num_clusters[self.file_name]}",
                                self.tensor_data)
                print('KMeans Clustered Amplitude calculation done')
                QApplication.restoreOverrideCursor()
            else:
                data = self.kmeans_dict[f"{self.file_name}_{self.num_clusters[self.file_name]}"]

                self.tensor_data = np.copy(data)
                uni = np.unique(self.tensor_data)
                print(uni)
                # After clustering, prompt the user to select which clusters to keep
                clusters_to_keep = self.select_clusters_to_keep(self.num_clusters[self.file_name])
                QApplication.setOverrideCursor(self.custom_cursor)
                if clusters_to_keep is not None:
                    print('keep')
                    # Set the values of the clusters not selected to zero
                    for i in range(1, self.num_clusters[self.file_name] + 1):
                        if i not in clusters_to_keep:
                            self.tensor_data[self.tensor_data[:, :, :, -1] == i] = 0

                uni = np.unique(self.tensor_data)
                print(uni)
                uni2 = np.unique(data)
                print('original', uni2)
                # Store the modified tensor with an informative key
                self.add_tensor(f"{self.file_name}_KMeans Clustered Amplitude {self.num_clusters[self.file_name]}",
                                self.tensor_data)
                print('KMeans Clustered Amplitude calculation done')
                QApplication.restoreOverrideCursor()
        else:
            print('KMeans Clustered Amplitude calculation aborted')

    def select_clusters_to_keep(self, num_clusters):
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Clusters to Keep')
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget(dialog)
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for i in range(1, num_clusters + 1):
            list_widget.addItem(f"Cluster {i}")
        layout.addWidget(list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # Define the styles for light and dark themes
        light_theme_style = """
        QDialog {
            background-color: #B2B2B2; /* Light grey background */
            color: #333333; /* Dark grey text */
        }
        /* Add other widget styles for light theme here */
        """
        # Define the dark theme style
        dark_theme_style = """
        QWidget {
        background-color: #333333; /* Dark grey background */
        color: #FFFFFF; /* White text for better readability */
        }
        /* Add other widget styles for dark theme here */
        """

        # Apply the appropriate style based on the theme
        if self.isDarkTheme:
            dialog.setStyleSheet(dark_theme_style)
        else:
            dialog.setStyleSheet(light_theme_style)

        result = dialog.exec_()
        if result == QDialog.Accepted:
            selected_items = [int(item.text().split()[-1]) for item in list_widget.selectedItems()]
            return selected_items
        return None

    # Function to update the menu item text based on the current view type
    def update_menu_action_text(self, current_index):
        if self.view_type.get(current_index) == '2D':
            self.toggle_canvas_action.setText("3D View")
        else:
            self.toggle_canvas_action.setText("2D View")

    def toggle_canvas_from_menu(self):
        current_index = self.tab_widget.currentIndex()
        if self.view_type[current_index] == '2D':
            self.toggle_canvas('3D')
        else:
            self.toggle_canvas('2D')

    def apply_grid_color(self):
        color = QColorDialog.getColor(options=QColorDialog.ShowAlphaChannel)
        if color.isValid():
            for active_tab_index in self.grid_color.keys():
                self.grid_color[active_tab_index] = color
                # Iterate over all keys in self.grids and set the color
                for grid_key in self.grids.keys():
                    self.grids[grid_key].setColor(color)

                # Iterate over all keys in self.grid_labels and set the color for each label in the list
                for label_key in self.grid_labels.keys():
                    for label in self.grid_labels[label_key]:
                        label.setData(color=color)

                # Iterate over all keys in self.grid_axis_labels and set the color for each axis label in the list
                for axis_label_key in self.grid_axis_labels.keys():
                    for axis_label in self.grid_axis_labels[axis_label_key]:
                        axis_label.setData(color=color)
            self.last_grid_color = color
            self.cglv.initial_grid_color()

    def set_grid_color(self):
        # Get the index of the currently active tab
        active_tab_index = self.tab_widget.currentIndex()

        # Check if the active tab index exists in the grids dictionary
        if active_tab_index in self.grids:
            self.grids[active_tab_index].setColor(self.grid_color[active_tab_index])

            for label in self.grid_labels[active_tab_index]:
                # Update the color of the x-axis label at the bottom
                label.setData(color=self.grid_color[active_tab_index])

            for axis_label in self.grid_axis_labels[active_tab_index]:
                axis_label.setData(color=self.grid_color[active_tab_index])
        else:
            # Handle the case where the active tab index is not in the grids dictionary
            print(f"No grid found for the active tab index: {active_tab_index}")

    def update_theme_initial_on_time(self):
        current_time = QTime.currentTime()
        # Handle dark mode times that span midnight
        if QTime(20, 0) < current_time or current_time < QTime(8, 0):
            self.isDarkTheme = True
            self.update_status_bar_color()
        else:
            self.isDarkTheme = False
            self.update_status_bar_color()

    def update_memory_usage(self):
        memory_info = psutil.virtual_memory()
        used_memory = memory_info.used / (1024 ** 3)  # Convert to GB
        total_memory = memory_info.total / (1024 ** 3)
        self.memory_label.setText(f"RAM: {used_memory:.2f} GB / {total_memory:.2f} GB")

        # Repeat the update every 2 seconds
        QTimer.singleShot(1000, self.update_memory_usage)

    def update_status_bar_color(self):
        if self.isDarkTheme:
            self.statusBar().setStyleSheet("background-color: #353535; color: Gainsboro;")
        else:
            self.statusBar().setStyleSheet("background-color: #B2B2B2; color: black;")

    def canvas_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.cglv.update_all_instances_background(color)
            g_clr = self.adjust_grid_color(color)
            for active_tab_index in self.grid_color.keys():
                self.grid_color[active_tab_index] = g_clr
                # Iterate over all keys in self.grids and set the color
                for grid_key in self.grids.keys():
                    self.grids[grid_key].setColor(g_clr)

                # Iterate over all keys in self.grid_labels and set the color for each label in the list
                for label_key in self.grid_labels.keys():
                    for label in self.grid_labels[label_key]:
                        label.setData(color=g_clr)

                # Iterate over all keys in self.grid_axis_labels and set the color for each axis label in the list
                for axis_label_key in self.grid_axis_labels.keys():
                    for axis_label in self.grid_axis_labels[axis_label_key]:
                        axis_label.setData(color=g_clr)
            self.last_grid_color = g_clr
            # Flag to track if any tab contains an instance of RoundedCanvas
            canvas_found = False

            # Iterate through all tabs
            for i in range(self.tab_widget.count()):
                # Retrieve the widget of the current tab
                tab_widget = self.tab_widget.widget(i)

                # Assuming the canvas is the first widget in the layout of the tab
                canvas_layout = tab_widget.layout()
                canvas = canvas_layout.itemAt(0).widget()

                if isinstance(canvas, PyQtGraphCanvas):
                    # Set the flag to True if an instance of RoundedCanvas is found
                    canvas_found = True
                    break  # Exit the loop since we only need to find one instance

            # Call the function if a RoundedCanvas instance is found
            if canvas_found:
                self.cglv.initial_grid_color()

            for i in range(self.tab_widget.count()):
                # Retrieve the widget of the current tab
                tab_widget = self.tab_widget.widget(i)

                # Assuming the canvas is the first widget in the layout of the tab
                canvas_layout = tab_widget.layout()
                canvas = canvas_layout.itemAt(0).widget()
                if isinstance(canvas, RoundedCanvas):
                    # Set the background color for the canvas
                    canvas.setBackground(color.name())
            self.last_selected_color = color
            self.cglv.background_color_loaded = color

    def toggle_visibility(self, active, *dicts):
        # Iterate through all dictionaries and their items
        for item_dict in dicts:
            for key, value in item_dict.items():
                # Check if the value is a list of items
                if isinstance(value, list):
                    # Iterate over each item in the list
                    for item in value:
                        # If the item is not None, set its visibility based on the 'active' parameter
                        if item is not None:
                            item.setVisible(active)
                else:
                    # If the value is not a list, it's a single item
                    # If the item is not None, set its visibility based on the 'active' parameter
                    if value is not None:
                        value.setVisible(active)

    def toggle_grid(self):
        self.grid_active = not self.grid_active
        # Set the text for the grid action based on its active state
        self.grid_action.setText("Deactivate Grid" if self.grid_active else "Activate Grid")
        # Toggle visibility for all non-None items in both dictionaries
        self.toggle_visibility(self.grid_active, self.grids_initial, self.grids, self.grid_labels,
                               self.grid_axis_labels, self.x_axis_label_bottom_initial, self.x_axis_label_top_initial,
                               self.y_axis_label_left_initial,
                               self.y_axis_label_right_initial)

    def choose_dark_mode_times(self):
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)

        # Create time edit widgets with 24-hour format
        start_time_edit = QTimeEdit(dialog)
        start_time_edit.setDisplayFormat("HH:mm")  # 24-hour format
        end_time_edit = QTimeEdit(dialog)
        end_time_edit.setDisplayFormat("HH:mm")  # 24-hour format

        layout.addWidget(start_time_edit)
        layout.addWidget(end_time_edit)

        # Create QDialogButtonBox for OK and Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setWindowTitle("Choose Dark Mode Schedule")

        # Execute the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            # Save the chosen times
            self.dark_mode_start_time = start_time_edit.time()
            self.dark_mode_end_time = end_time_edit.time()
            # Update the theme based on the chosen times
            self.automatic_theme_enabled = True
            self.update_theme_based_on_time()
            self.update_status_bar_color()

    def choose_background_color(self):
        # Open a QColorDialog to choose a color
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"QMainWindow {{background-color: {color.name()};}}")
            # Disable automatic theme switching
            self.automatic_theme_enabled = False

    def choose_background_color_canvas(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.cglv.setBackgroundColor(color)

    def choose_menu_bar_color(self):
        # Open a QColorDialog to choose a color
        color = QColorDialog.getColor()
        if color.isValid():
            self.menubar.setStyleSheet(f"QMenuBar {{background-color: {color.name()};}}")
            # Disable automatic theme switching
            self.automatic_theme_enabled = False

    def update_theme_based_on_time(self):
        if self.automatic_theme_enabled:
            current_time = QTime.currentTime()
            # Handle dark mode times that span midnight
            if ((self.dark_mode_start_time < self.dark_mode_end_time and
                 current_time >= self.dark_mode_start_time and
                 current_time < self.dark_mode_end_time) or
                    (self.dark_mode_start_time > self.dark_mode_end_time and
                     (current_time >= self.dark_mode_start_time or
                      current_time < self.dark_mode_end_time))):
                self.apply_dark_theme()
                self.isDarkTheme = True
                self.update_status_bar_color()
            else:
                self.isDarkTheme = False
                self.update_status_bar_color()
                self.apply_light_theme()

    def set_menu_color(self, label, color):
        label.setStyleSheet(f"QMenuBar {{ color : {color}; }}")

    def apply_light_theme(self):
        # Set the QMenuBar style with a light color scheme
        self.setStyleSheet("""
        QMainWindow {
            background-color: #B2B2B2;
        }
        QPushButton {
            background-color: #EDEDED;
            color: 333333;
        }
        QPushButton:hover {
            background-color: #B2B2B2;
        }
        QMenuBar {
            background-color: #CACACA; /* Light grey */
            color: #333333; /* Dark grey for text */
        }
        QMenuBar::item:selected {
            background-color: #B2B2B2; /* Slightly darker grey for selected item */
            color: #333333;
        }
        QMenu {
            background-color: #F0F0F0; /* Light grey background for the menu */
            color: #333333; /* Dark grey text for better readability */
        }
        QMenu::item {
            background-color: #F0F0F0; /* Light grey background for each item */
            color: #333333; /* Dark grey text for better readability */
        }
        QMenu::item:selected {
            background-color: #B2B2B2; /* Slightly darker grey for selected item */
            color: #333333; /* Keeping the text color consistent with the theme */
        }
        QTreeWidget {
            background-color: #F0F0F0; /* Light grey background for the tree widget */
            color: #333333; /* Dark grey text for better readability */
        }

        QTreeWidget::item {
            background-color: #F0F0F0; /* Light grey background for each item */
            color: #333333; /* Dark grey text for better readability */
        }

        QTreeWidget::item:selected {
            background-color: #CACACA; /* Slightly darker grey for selected item */
            color: #333333; /* Keeping the text color consistent with the theme */
        }

        QHeaderView::section {
            background-color: #E8E8E8; /* Lighter grey for header sections */
            color: #333333; /* Dark grey text for better readability */
        }

        QTreeWidget QMenu {
            background-color: #F0F0F0; /* Light grey background for the context menu */
            color: #333333; /* Dark grey text for better readability */
        }

        QTreeWidget QMenu::item:selected {
            background-color: #CACACA; /* Slightly darker grey for selected context menu item */
            color: #333333; /* Keeping the text color consistent with the theme */
        }
        QAbstractItemView {
            selection-background-color: #CACACA;
            selection-color: 333333;
        }
        QTabWidget::pane {
            background-color: #EDEDED; /* Choose from the pane background color options */
            border-style: solid;
            border-color: #EDEDED; /* Choose from 
            border-width: 0 1px 1px 1px; /* Top, Right, Bottom, Left */
        }

        QTabBar::tab:selected {
            background: #EDEDED; /* Lighter background for selected tab */
            border-style: solid;
            border-color: #EDEDED; /* Lighter border color */
            border-width: 1px 1px 0 1px; /* Top, Right, Bottom, Left */
            border-top-left-radius: 3px; /* Rounds the top-left corner */
            border-top-right-radius: 3px; /* Rounds the top-right corner */
            padding: 4px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); /* Softer shadow for light theme */
        }

        QTabBar::tab {
            background: #D9D9D9; /* Lighter background for tabs */
            color: #333333; /* Darker text color for better readability */
            border-style: solid;
            border-color: #D9D9D9; /* Lighter border color */
            border-width: 1px 1px 0 1px; /* Top, Right, Bottom, Left */
            padding: 4px;
            border-top-left-radius: 3px; /* Rounds the top-left corner */
            border-top-right-radius: 3px; /* Rounds the top-right corner */
            box-shadow: none;
        }

        QTabBar::scroller { /* The scroller sub-control */
            background: #E8E8E8; /* Lighter background color for the scroller */
        }

        QTabBar QToolButton { /* Targeting the scroll buttons specifically */
            background: #E8E8E8; /* Lighter background color for the buttons */
            border: none; /* Optional: removes the border */
            padding: 3px;
        }

        QTabBar QToolButton:hover {
            background-color: #B2B2B2; /* Slightly darker background color on hover */
            color: #333333; /* Darker text color on hover */
            /* Additional styling properties as needed */
        }

        QTabBar::tear {
            background: #F0F0F0; /* Lighter background color for the tear */
            }

         /* Style for QSlider groove */
        QSlider::groove:horizontal {
            border: 0px solid #999999;
            height: 1px; /* Set the height of the groove */
            background: #E8E8E8; /* Light grey background for the unfilled part */
            margin: 1px 0;
            border-radius: 5px; /* Rounded corners for the handle */
        }

        /* Style for QSlider handle */
        QSlider::handle:horizontal {
            background: #353535; /* Slightly darker grey for the handle */
            border: 0px solid #E8E8E8; /* Border color for the handle */
            height: 18px; /* Taller handle */
            width: 10px; /* Slimmer handle */
            margin: -5px 0; /* Expand outside the groove */
            border-radius: 3px; /* Rounded corners for the handle */
        }

        /* Style for QSlider filled part */
        QSlider::add-page:horizontal {
            background: #CACACA; /* Slightly darker grey for the filled part */
        }

        /* Style for QSlider unfilled part */
            QSlider::sub-page:horizontal {
            background: #353535; /* Light grey background for the unfilled part */
        }   

        QProgressBar {
            border: 2px solid #5C5C5C; /* Light grey border */
            border-radius: 5px; /* Rounded corners for the progress bar */
            background-color: #5C5C5C; /* Light grey background */
            text-align: center; /* Center the text */
        }

        QProgressBar::chunk {
            background-color: #5C5C5C; /* Slightly darker grey for the progress */
        }
        QLineEdit {
            background-color: #F0F0F0;
            color: 333333;
            border-radius: 2px;
            height: 20px;
        }
        QLineEdit:focus { 
            border: 1px solid #5C5C5C; 
            border-radius: 2px; /* Rounded corners for the progress bar */
        }
        QDialog {
            background-color: #B2B2B2;
        }
        QScrollArea {
            background-color: #F0F0F0; /* Light grey background for the scroll area */
            color: #333333; /* Dark grey text for better readability */
        }
        
        QScrollArea QWidget {
            background-color: #F0F0F0; /* Light grey background for the content inside the scroll area */
            color: #333333; /* Dark grey text for better readability */
        }

        """)

    def apply_dark_theme(self):
        # Set the dark theme styles
        style_sheet = """
        QMainWindow {
            background-color: #353535;
        }
        QMenuBar {
            background-color: #404040;
            color: #FFFFFF;
        }
        QMenuBar::item:selected {
            background-color: #5C5C5C;
            color: white;
        }
        QPushButton {
            background-color: #404040;
            color: white;
        }
        QPushButton:hover {
            background-color: #5C5C5C;
        }
        QLabel {
            color: Gainsboro;
        }
        QMenu {
            background-color: #404040;
            color: #FFFFFF;
        }
        QMenu::item {
            background-color: #404040;
            color: #FFFFFF;
        }
        QMenu::item:selected {
            background-color: #5C5C5C;
        }
        QTabWidget::pane {
            background-color: #404040; /* Choose from the pane background color options */
            border-style: solid;
            border-color: #404040;
            border-width: 0 1px 1px 1px; /* Top, Right, Bottom, Left */
        }
        QTabBar::tab:selected {
            background: #404040;
            border-style: solid;
            border-color: #404040;
            border-width: 1px 1px 0 1px; /* Top, Right, Bottom, Left */
            border-top-left-radius: 3px; /* Rounds the top-left corner */
            border-top-right-radius: 3px; /* Rounds the top-right corner */
            padding: 4px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }
        QTabBar::tab {
            background: #5C5C5C;
            color: white;
            border-style: solid;
            border-color: #5C5C5C;
            border-width: 1px 1px 0 1px; /* Top, Right, Bottom, Left */
            padding: 4px;
            border-top-left-radius: 3px; /* Rounds the top-left corner */
            border-top-right-radius: 3px; /* Rounds the top-right corner */
            box-shadow: none;
        }
        QTabBar::scroller { /* The scroller sub-control */
            background: #404040; /* Change the color as desired */
        }
        QTabBar QToolButton { /* Targeting the scroll buttons specifically */
            background: #404040; /* Change the color as desired */
            border: solid; /* Optional: removes the border */
            padding: 3px;
        }
        QTabBar QToolButton:hover {
            background-color: #555; /* Change to your desired hover background color */
            color: #fff; /* Change to your desired hover text color */
            /* Additional styling properties as needed */
        }
        QTabBar::tear {
            background: #404040; /* Set your desired color */
        }
        QSpinBox, QDateTimeEdit {
            background-color: #353535;
            color: white;
        }
        QLineEdit {
            background-color: #5C5C5C;
            color: white;
            border-radius: 2px;
            height: 20px;
        }
        QLineEdit:focus { 
            border: 1px solid #B2B2B2; 
            border-radius: 2px; /* Rounded corners for the progress bar */
        }
        QTreeWidget {
        background-color: #404040;
        color: white;
        }
        QTreeWidget::item {
            background-color: #404040;
            color: white;
        }
        QTreeWidget::item:selected {
            background-color: #5C5C5C;
            color: white;
        }
        QHeaderView::section {
            background-color: #404040;
            color: white;
        }
        QTreeWidget QMenu {
            background-color: #404040;
            color: white;
        }
        QTreeWidget QMenu::item:selected {
            background-color: #5C5C5C;
            color: white;
        }
        QDialog {
        background-color: #353535;
        }
        QAbstractItemView {
            selection-background-color: #5C5C5C;
            selection-color: white;
        }
        /* Style for QSlider groove */
        QSlider::groove:horizontal {
            border: 0px solid #999999;
            height: 1px; /* Set the height of the groove */
            background: #E8E8E8; /* Light grey background for the unfilled part */
            margin: 1px 0;
            border-radius: 5px; /* Rounded corners for the handle */
        }

        /* Style for QSlider handle */
        QSlider::handle:horizontal {
            background: #B2B2B2; /* Slightly darker grey for the handle */
            border: 0px solid #5C5C5C; /* Border color for the handle */
            height: 18px; /* Taller handle */
            width: 10px; /* Slimmer handle */
            margin: -5px 0; /* Expand outside the groove */
            border-radius: 3px; /* Rounded corners for the handle */
        }

        /* Style for QSlider filled part */
        QSlider::add-page:horizontal {
            background: #5C5C5C; /* Slightly darker grey for the filled part */
        }

        /* Style for QSlider unfilled part */
            QSlider::sub-page:horizontal {
            background: #B2B2B2; /* Light grey background for the unfilled part */
        }

        QProgressBar {
            border: 2px solid #5C5C5C; /* Light grey border */
            border-radius: 5px; /* Rounded corners for the progress bar */
            background-color: #5C5C5C; /* Light grey background */
            text-align: center; /* Center the text */
        }

        QProgressBar::chunk {
            background-color: #5C5C5C; /* Slightly darker grey for the progress */
        }
        
        QScrollArea {
            background-color: #404040; /* Dark grey background for the scroll area */
            color: white; /* White text for better readability */
        }
        
        QScrollArea QWidget {
            background-color: #404040; /* Dark grey background for the content inside the scroll area */
            color: white; /* White text for better readability */
        }

        """
        self.setStyleSheet(style_sheet)

    def update_recent_files_submenu(self):
        # Clear existing actions
        self.recent_files_menu.clear()

        # Add actions for each loaded file
        for file_name in self.loaded_file_paths:
            recent_file_action = QAction(file_name, self)
            recent_file_action.triggered.connect(lambda _, name=file_name: self.load_recent_file(name))
            self.recent_files_menu.addAction(recent_file_action)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected_action = self.recent_files_menu.activeAction()
            if selected_action:
                file_name = selected_action.text()
                # Remove the file from your data structure (e.g., loaded_file_paths)
                if file_name in self.loaded_file_paths:
                    del self.loaded_file_paths[file_name]  # Remove the file
                    self.update_recent_files_submenu()  # Re-populate the submenu after deletion
            else:
                selected_items = self.treeWidget.selectedItems()
                if selected_items:
                    selected_item = selected_items[0]
                    parent_item = selected_item.parent()
                    is_parent = parent_item is None

                    # Determine the key based on whether the selected item is a parent or a child
                    key = selected_item.text(
                        0).strip() if is_parent else f"{parent_item.text(0).strip()} {selected_item.text(0).strip()}"

                    # Perform deletion from custom data structures
                    if is_parent:
                        if self.inst_phase_radians.get(self.file_name) is not None:
                            del self.inst_phase_radians[key]
                        if self.analytic_signal.get(self.file_name) is not None:
                            del self.analytic_signal[key]
                        if self.envelope_dict.get(self.file_name) is not None:
                            del self.envelope_dict[key]
                        if self.inst_freq_hz_padded.get(self.file_name) is not None:
                            del self.inst_freq_hz_padded[key]
                        if self.inst_bandwidth_dict.get(self.file_name) is not None:
                            del self.inst_bandwidth_dict[key]
                        if self.num_clusters.get(self.file_name) is not None:
                            if self.kmeans_dict.get(
                                    f"{self.file_name}_{self.num_clusters[self.file_name]}") is not None:
                                # Define the reference key
                                reference_key = key
                                # Create a list to store keys to delete
                                keys_to_delete = []

                                # Iterate over the keys in the dictionary
                                for key in self.kmeans_dict.keys():
                                    # Split the key into parts using "_" as the delimiter
                                    key_parts = key.split("_")
                                    # Check if the first part of the key matches the reference key
                                    if key_parts[0] == reference_key:
                                        # If it does, add the key to the list of keys to delete
                                        keys_to_delete.append(key)

                                # Delete the items associated with the keys found
                                for key in keys_to_delete:
                                    del self.kmeans_dict[key]

                            # Remove all child tensors associated with this parent
                            child_keys_to_delete = [child_key for child_key in self.tensor_dict.keys() if
                                                    child_key.startswith(key + ' ')]
                            for child_key in child_keys_to_delete:
                                del self.tensor_dict[child_key]

                    # Remove the tensor from the dictionary
                    if key in self.tensor_dict:
                        del self.tensor_dict[key]

                    # If it's a child item, remove it from the parent
                    if not is_parent:
                        parent_item.removeChild(selected_item)

                    # Remove the selected item from the tree
                    index_of_deleted = self.treeWidget.indexOfTopLevelItem(
                        selected_item) if is_parent else parent_item.indexOfChild(selected_item)
                    if is_parent:
                        self.treeWidget.takeTopLevelItem(index_of_deleted)
                    else:
                        parent_item.takeChild(index_of_deleted)

                    # Handle the selection and setting of the new item
                    new_selected_item = None
                    if self.treeWidget.topLevelItemCount() > 0:
                        if index_of_deleted == 0:
                            new_selected_item = self.treeWidget.topLevelItem(
                                0)  # Select the first item if the deleted was the first
                        else:
                            new_selected_item = self.treeWidget.topLevelItem(
                                index_of_deleted - 1)  # Select the previous item if the deleted was not the first
                    else:
                        # If no items are left, clear the selection and data
                        self.tensor_data = None
                        self.file_name = None
                        self.file_path = None
                        self.three_d()
                        self.index_entry.setText("0")
                        # Clear the tensor dictionary
                        for key in list(self.tensor_dict.keys()):
                            del self.tensor_dict[key]

                    if new_selected_item:
                        self.treeWidget.setCurrentItem(new_selected_item)
                        new_key = new_selected_item.text(0).strip()
                        self.selectTensor(new_key)  # Call the selectTensor function with the new key

    def load_recent_file(self, file_name):
        if file_name in self.loaded_file_paths:
            file_path = self.loaded_file_paths[file_name]
            print(f"Loading recent file: {file_path}")
            # Check if the file exists
            if not file_path:
                print("No file selected, operation canceled.")
                return
            else:
                self.three_d()
                self.file_entry.setText(file_path)
                self.index_entry.setText("0")
                # Add the loaded file path to the dictionary
                file_name = file_path.split("/")[-1]  # Extract the file name from the path

                if '_' in file_name:
                    self.file_name = file_name.split('_', 1)[0]
                else:
                    self.file_name = file_name.split('.', 1)[0]

                self.loaded_file_paths[self.file_name] = file_path
                self.file_path = file_path
                self.update_recent_files_submenu()  # Update the recent files submenu
                try:
                    if self.file_path.endswith('.npy'):
                        QApplication.setOverrideCursor(self.custom_cursor)
                        print(f"Attempting to load tensor from: {file_path}")
                        self.progress_bar.show()
                        # Simulate data loading progress (replace with your actual loading logic)
                        total_bytes = os.path.getsize(file_path)
                        loaded_bytes = 0

                        with open(file_path, "rb") as file:
                            chunk_size = 512000000  # Adjust as needed
                            while True:
                                chunk = file.read(chunk_size)
                                if not chunk:
                                    break
                                loaded_bytes += len(chunk)
                                progress = int(loaded_bytes / total_bytes * 100)
                                self.progress_bar.setValue(progress)
                                QApplication.processEvents()  # Update the UI
                        print(file_path)
                        self.tensor_data = np.load(file_path)
                        # Add the loaded tensor to the dictionary
                        self.add_tensor(self.file_name, self.tensor_data)
                        print(f"Successfully loaded tensor. Shape: {self.tensor_data.shape}")
                        self.show_tensor_info()
                        self.progress_bar.hide()
                        self.channel_index_entry.setText("0")
                        if self.tensor_data.shape[-1] > 1:
                            self.set_slider_range(self.tensor_data.shape[-1])
                            self.channel_index_label.show()
                            self.channel_index_entry.show()
                            self.Channel_slider.show()
                        else:
                            self.channel_index_label.hide()
                            self.channel_index_entry.hide()
                            self.Channel_slider.hide()
                        # Check if the loaded tensor has more than four dimension
                        if len(self.tensor_data.shape) > 4:
                            self.channel_label.show()
                            self.channel_entry.show()
                        else:
                            self.channel_label.hide()
                            self.channel_entry.hide()
                        sampling_interval = self.get_sampling_interval_from_file_name()
                        current_index = self.tab_widget.currentIndex()
                        current_widget = self.tab_widget.widget(current_index)
                        if current_widget.layout():
                            canvas = current_widget.layout().itemAt(0).widget()

                            if isinstance(canvas, PyQtGraphCanvas):
                                # If the PyQtGraphCanvas is shown, hide the sampling interval UI elements
                                self.sampling_interval_label.hide()
                                self.sampling_interval_entry.hide()
                            elif isinstance(canvas, RoundedCanvas) and sampling_interval is None:
                                # If the RoundedCanvas is shown and there is no sampling interval, show the UI elements
                                self.sampling_interval_label.show()
                                self.sampling_interval_entry.show()

                        if not sampling_interval is None:
                            self.sampling_interval_label.hide()
                            self.sampling_interval_entry.hide()

                        QApplication.restoreOverrideCursor()
                    else:
                        # Extract the display name (name without extension)
                        display_name = os.path.splitext(os.path.basename(self.file_path))[0]

                        # Parse the LAS file
                        csv_data = self.parse_las_file(self.file_path)

                        # Store the data in the tensor dictionary with the display name as the key
                        self.tensor_dict[display_name] = csv_data
                        self.loaded_file_paths[display_name] = self.file_path

                        # Use existing function to add the entry to the tree widget
                        self.add_tensor(display_name, csv_data)

                except Exception as e:
                    QApplication.restoreOverrideCursor()
                    import traceback
                    traceback.print_exc()
                    QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            print(f"File '{file_name}' not found in loaded files.")

    def enter_file_name(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Enter File Name", "", "All Files (*)")
        if file_name:
            # User selected a valid file name
            self.save_file_name = file_name
        else:
            # User canceled, revert to default name
            self.save_file_name = None

    def closeEvent(self, event):
        # Generate a custom message box when clicking the 'X' button in the title bar
        reply = CustomMessageBox(theme_is_dark=self.isDarkTheme,
                                 text="Are you sure you want to exit?",
                                 icon=QMessageBox.Question,
                                 buttons=QMessageBox.Yes | QMessageBox.No,
                                 default_button=QMessageBox.No,
                                 parent=self)

        reply.setWindowTitle("Exit Confirmation")  # Set the title

        if reply.exec_() == QMessageBox.Yes:
            self.save_recent_files()  # Save the recent files before closing
            self.cglv.save_background_color(self.cglv.background_color_loaded)
            # Clean exit
            QApplication.quit()
        else:
            # Cancel the exit
            event.ignore()

    def save_recent_files(self):
        # Save the loaded_file_paths dictionary to a JSON file
        with open(self.recent_files_path, 'w') as file:
            json.dump(self.loaded_file_paths, file, indent=4)

    def load_recent_files(self):
        # Load the recent files from a JSON file into the loaded_file_paths dictionary
        if os.path.exists(self.recent_files_path):
            with open(self.recent_files_path, 'r') as file:
                file_paths = json.load(file)
            # Clear the current dictionary to avoid duplicates
            self.loaded_file_paths.clear()
            # Update the dictionary with file names as keys and full paths as values
            for file_name, file_path in file_paths.items():
                self.loaded_file_paths[file_name] = file_path

    def show_context_menu2(self, position):
        sender = self.sender()
        context_menu = QMenu(self)

        if isinstance(sender, RoundedCanvas):
            # Get the index of the currently active tab
            active_tab_index = self.tab_widget.currentIndex()

            switch_to_3d_action = context_menu.addAction("Switch to 3D")
            # New action to choose a color
            save_action = context_menu.addAction("Save")

            choose_color_action = context_menu.addAction("Choose Background Color")
            choose_color_action.triggered.connect(self.choose_color)

            if self.cbar.get(active_tab_index) is not None:
                # Call the function to add the color mapping submenu
                self.addCustomColorMappingSubMenu(context_menu, self.cbar[active_tab_index])

            action = context_menu.exec_(sender.mapToGlobal(position))

            if action == switch_to_3d_action:
                self.toggle_canvas('3D')
            elif action == save_action:
                # Save the plot using PyQtGraph's exporter
                # Get the index of the currently active tab
                active_tab_index = self.tab_widget.currentIndex()
                # Retrieve the widget of the currently active tab
                active_tab_widget = self.tab_widget.widget(active_tab_index)
                # Assuming the canvas is the first widget in the layout of the active tab
                canvas_layout = active_tab_widget.layout()
                canvas = canvas_layout.itemAt(0).widget()
                self.export_dialog = exportDialog.ExportDialog(canvas.scene())
                self.export_dialog.show(canvas.plotItem)

    # Function to apply the LUT to the colorBarItem
    def applyLUT(self, colorBarItem, colorMap):
        colorBarItem.setColorMap(colorMap)

    # Function to add custom color mapping options to the context menu
    def addCustomColorMappingSubMenu(self, contextMenu, colorBarItem):
        # Define your custom color mappings here
        customColorMappings = {
            'Seismic Default': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
            'Polarity': ['#000080', 'white', '#BF0000'],
            'Frequency': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
            'Cosine of Phase': ['black', 'white'],
            'Ins. Phase': ['#FF69B4', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF']
        }

        # Create the custom color mapping submenu
        customColorMapMenu = contextMenu.addMenu('Custom Color Mapping')

        # Populate the submenu with custom color map options
        for cmapName, colors in customColorMappings.items():
            action = customColorMapMenu.addAction(cmapName)
            action.triggered.connect(lambda checked, colors=colors: self.createAndApplyLUT(colorBarItem, colors))

    # Function to create a LUT from a list of colors and apply it to the colorBarItem
    def createAndApplyLUT(self, colorBarItem, colors):
        # Calculate the number of colors
        num_colors = len(colors)

        # Create a list of positions for the color stops, ranging from 0 to 1
        # The middle stop should be at 0.5, representing zero
        stops = [(i / (num_colors - 1)) for i in range(num_colors)]

        # Create the ColorMap object with symmetrical stops around the middle color
        color_map = pg.ColorMap(stops, [pg.mkColor(c) for c in colors])
        # Apply the LUT to the colorBarItem
        self.applyLUT(colorBarItem, color_map)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Get the index of the currently active tab
            active_tab_index = self.tab_widget.currentIndex()
            # Retrieve the widget of the currently active tab
            active_tab_widget = self.tab_widget.widget(active_tab_index)
            # Assuming the canvas is the first widget in the layout of the active tab
            canvas_layout = active_tab_widget.layout()
            canvas = canvas_layout.itemAt(0).widget()
            # Assuming you want to set the color for the RoundedCanvas background
            canvas.setBackground(color.name())

    def three_d(self):
        self.dim1_entry.clear()
        self.dim2_entry.clear()
        self.index_dim_entry.clear()
        self.index_entry.setText("0")
        self.Three_D_button.setChecked(True)
        self.index_label.hide()
        self.index_entry.hide()
        self.index_slider.hide()

    def toggle_canvas(self, view_type):
        current_index = self.tab_widget.currentIndex()
        self.tab_widget.currentChanged.connect(self.on_tab_change)
        current_widget = self.tab_widget.widget(current_index)
        if isinstance(current_widget, QWidget):
            current_layout = current_widget.layout()
            if current_layout:
                canvas = current_layout.itemAt(0).widget()
                if view_type == '3D' and isinstance(canvas, RoundedCanvas):
                    # Switch to OpenGL canvas
                    opengl_canvas = PyQtGraphCanvas(self)
                    current_layout.addWidget(opengl_canvas)
                    current_layout.removeWidget(canvas)
                    canvas.deleteLater()  # Remove the previous canvas from memory
                    self.tab_volume_items.pop(current_index, None)
                    # Set the background color for the new tab based on the most recent selection
                    if self.custom_gl_widgets:
                        new_color = self.last_selected_color
                    else:
                        # Default color if no tabs have been updated yet
                        new_color = self.cglv.background_color_loaded

                    self.custom_gl_widgets[current_index] = opengl_canvas
                    opengl_canvas.plot_widget.setBackgroundColor(new_color)
                    # Update the dictionaries with references to the new canvas items
                    self.grid_color[current_index] = self.last_grid_color
                    self.view_type[current_index] = '3D'
                    self.x_axis_label_bottom_initial[current_index] = opengl_canvas.plot_widget.x_axis_label_bottom
                    self.x_axis_label_top_initial[current_index] = opengl_canvas.plot_widget.x_axis_label_top
                    self.y_axis_label_left_initial[current_index] = opengl_canvas.plot_widget.y_axis_label_left
                    self.y_axis_label_right_initial[current_index] = opengl_canvas.plot_widget.y_axis_label_right
                    self.grids_initial[current_index] = opengl_canvas.plot_widget.grid
                    self.canvas = opengl_canvas  # Update self.canvas to refer to the new canvas
                    self.plot_button.clicked.disconnect()  # Disconnect previous signal
                    self.plot_button.clicked.connect(self.plot_data3d)  # Connect to 3D plot function
                    self.tab_3D_state[current_index] = False  # Initialize the state as no 3D plot displayed
                    self.sampling_interval_label.hide()
                    self.sampling_interval_entry.hide()
                    self.Three_D_button.show()
                    self.three_d()
                    self.update_active_tab_name()
                    self.update_menu_action_text(current_index)
                    if current_index in self.grids:
                        del self.grid_axis_labels[current_index]
                        del self.grid_labels[current_index]
                        del self.grids[current_index]
                    if self.file_path.endswith('.las'):
                        self.sampling_interval_label.hide()
                        self.sampling_interval_entry.hide()
                        self.time_slice_button.hide()
                        self.cross_line_button.hide()
                        self.inline_button.hide()
                        self.channel_index_entry.hide()
                        self.channel_index_label.hide()
                        self.Channel_slider.hide()
                        self.Three_D_button.hide()
                        self.plot_button.clicked.disconnect()  # Disconnect previous signal
                        self.plot_button.clicked.connect(self.well_log_viewer)
                elif view_type == '2D' and isinstance(canvas, PyQtGraphCanvas):
                    # Switch to rounded canvas
                    rounded_canvas = RoundedCanvas(self)
                    current_layout.addWidget(rounded_canvas)
                    current_layout.removeWidget(canvas)
                    canvas.deleteLater()  # Remove the previous canvas from memory
                    self.custom_gl_widgets.pop(current_index, None)
                    self.plot_button.clicked.disconnect()  # Disconnect previous signal
                    self.plot_button.clicked.connect(self.plot_tensor)
                    self.Three_D_button.hide()
                    self.three_d()
                    self.cbar[current_index] = None
                    self.view_type[current_index] = '2D'
                    self.grid_color[current_index] = self.last_grid_color
                    del self.grids_initial[current_index]
                    if current_index in self.grids:
                        del self.grids[current_index]
                    del self.x_axis_label_bottom_initial[current_index]
                    del self.x_axis_label_top_initial[current_index]
                    del self.y_axis_label_left_initial[current_index]
                    del self.y_axis_label_right_initial[current_index]
                    if current_index in self.grid_labels:
                        del self.grid_labels[current_index]
                    if current_index in self.grid_axis_labels:
                        del self.grid_axis_labels[current_index]
                    rounded_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
                    rounded_canvas.customContextMenuRequested.connect(self.show_context_menu2)
                    self.update_active_tab_name()
                    self.update_menu_action_text(current_index)
                    # Check if the loaded tensor has sampling_interval
                    sampling_interval = self.get_sampling_interval_from_file_name()
                    if sampling_interval is None and self.tensor_data is not None:
                        self.sampling_interval_label.show()
                        self.sampling_interval_entry.show()
                    if self.file_path.endswith('.las'):
                        self.sampling_interval_label.hide()
                        self.sampling_interval_entry.hide()
                        self.time_slice_button.hide()
                        self.cross_line_button.hide()
                        self.inline_button.hide()
                        self.channel_index_entry.hide()
                        self.channel_index_label.hide()
                        self.Channel_slider.hide()
                        self.Three_D_button.hide()
                        self.plot_button.clicked.disconnect()  # Disconnect previous signal
                        self.plot_button.clicked.connect(self.well_log_viewer)

    def add_tab(self):
        new_tab = QWidget()  # Create a new tab widget
        new_tab_layout = QVBoxLayout(new_tab)  # Create a layout for the new tab
        current_index = self.tab_widget.currentIndex()
        current_widget = self.tab_widget.widget(current_index)
        new_tab_index = self.tab_widget.count()  # Index for the new tab
        # Set the background color for the new tab based on the most recent selection
        if self.custom_gl_widgets:
            new_color = self.last_selected_color
        else:
            # Default color if no tabs have been updated yet
            new_color = self.cglv.background_color_loaded

        if isinstance(current_widget, QWidget):
            current_layout = current_widget.layout()
            if current_layout:
                canvas = current_layout.itemAt(0).widget()
                if isinstance(canvas, RoundedCanvas):
                    new_canvas = RoundedCanvas(self)
                    tab_name = "2D Plot"  # Name for tabs with RoundedCanvas
                    new_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
                    new_canvas.customContextMenuRequested.connect(self.show_context_menu2)
                    self.view_type[new_tab_index] = '2D'
                    self.cbar[new_tab_index] = None
                elif isinstance(canvas, PyQtGraphCanvas):
                    new_canvas = PyQtGraphCanvas(self)
                    # Update the dictionary with the new CustomGLViewWidget
                    self.custom_gl_widgets[new_tab_index] = new_canvas
                    new_canvas.plot_widget.setBackgroundColor(new_color)
                    tab_name = "3D Plot"  # Name for tabs with PyQtGraphCanvas
                    self.view_type[new_tab_index] = '3D'
                    self.tab_3D_state[new_tab_index] = False  # Initialize the state as no 3D plot displayed
                new_tab_layout.addWidget(new_canvas)
                # Add the new tab to the tab widget with the next sequential number
                self.tab_widget.addTab(new_tab, f"{tab_name} {self.tab_widget.count() + 1} ")

                if isinstance(new_canvas, PyQtGraphCanvas):
                    self.grid_color[new_tab_index] = self.last_grid_color
                    self.grids_initial[new_tab_index] = new_canvas.plot_widget.grid
                    self.x_axis_label_bottom_initial[new_tab_index] = new_canvas.plot_widget.x_axis_label_bottom
                    self.x_axis_label_top_initial[new_tab_index] = new_canvas.plot_widget.x_axis_label_top
                    self.y_axis_label_left_initial[new_tab_index] = new_canvas.plot_widget.y_axis_label_left
                    self.y_axis_label_right_initial[new_tab_index] = new_canvas.plot_widget.y_axis_label_right

                self.update_tab_name()  # Update all tab names to be sequential
                self.update_active_tab_name()

        else:
            print("Current widget is not found or invalid.")

    def update_tab_name(self):
        # Update the names of all tabs to be sequential
        for index in range(self.tab_widget.count()):
            current_tab_name = self.tab_widget.tabText(index)
            tab_type = current_tab_name.split()[0]  # Extract the type (2D or 3D) from the current tab name
            # Update the tab name with the new sequential number
            self.tab_widget.setTabText(index, f"{tab_type} Plot {index + 1} ")

    def update_active_tab_name(self):
        current_index = self.tab_widget.currentIndex()
        current_tab_name = self.tab_widget.tabText(current_index)
        tab_number = current_tab_name.split()[-1]  # Extract the number from the current tab name

        # Get the current widget in the tab
        current_widget = self.tab_widget.widget(current_index)
        if current_widget.layout():
            # Get the first widget in the layout, which should be the canvas
            canvas = current_widget.layout().itemAt(0).widget()
            if isinstance(canvas, PyQtGraphCanvas):
                # If the current canvas is PyQtGraphCanvas, it's 3D
                self.tab_widget.setTabText(current_index, f"3D Plot {tab_number} ")
            elif isinstance(canvas, RoundedCanvas):
                # If the current canvas is RoundedCanvas, it's 2D
                self.tab_widget.setTabText(current_index, f"2D Plot {tab_number} ")

    def on_tab_change(self, current_index):
        current_widget = self.tab_widget.widget(current_index)
        if isinstance(current_widget, QWidget):
            current_layout = current_widget.layout()
            if current_layout:
                canvas = current_layout.itemAt(0).widget()
                # Update connections for the plot button
                self.plot_button.clicked.disconnect()
                # Connect to the appropriate plotting function based on the type of canvas
                if isinstance(canvas, PyQtGraphCanvas):
                    self.plot_button.clicked.connect(self.plot_data3d)
                    self.sampling_interval_label.hide()
                    self.sampling_interval_entry.hide()
                    self.Three_D_button.show()
                    self.three_d()
                    self.update_menu_action_text(current_index)
                elif isinstance(canvas, RoundedCanvas):
                    self.plot_button.clicked.connect(self.plot_tensor)
                    self.Three_D_button.hide()
                    self.three_d()
                    self.update_menu_action_text(current_index)
                    # Check if the loaded tensor has sampling_interval
                    sampling_interval = self.get_sampling_interval_from_file_name()
                    if sampling_interval is None and self.tensor_data is not None:
                        self.sampling_interval_label.show()
                        self.sampling_interval_entry.show()
                if self.file_path.endswith('.las'):
                    self.sampling_interval_label.hide()
                    self.sampling_interval_entry.hide()
                    self.time_slice_button.hide()
                    self.cross_line_button.hide()
                    self.inline_button.hide()
                    self.channel_index_entry.hide()
                    self.channel_index_label.hide()
                    self.Channel_slider.hide()
                    self.Three_D_button.hide()
                    self.plot_button.clicked.disconnect()  # Disconnect previous signal
                    self.plot_button.clicked.connect(self.well_log_viewer)

    def close_tab(self, closed_tab_index):
        # Remove the tab at closed_tab_index
        self.tab_widget.removeTab(closed_tab_index)

        # Update dictionaries for remaining tabs
        for dictionary in [self.grids, self.grids_initial,
                           self.tab_volume_items, self.custom_gl_widgets, self.grid_axis_labels, self.grid_labels,
                           self.x_axis_label_bottom_initial, self.x_axis_label_top_initial,
                           self.y_axis_label_left_initial,
                           self.y_axis_label_right_initial, self.grid_color, self.view_type, self.tab_3D_state,
                           self.cbar]:
            # Create a new dictionary to store updated keys and values
            updated_dict = {}
            for key, value in dictionary.items():
                # Decrement the keys by 1 for tabs after the closed tab
                if key > closed_tab_index:
                    updated_dict[key - 1] = dictionary[key]
                elif key < closed_tab_index:
                    updated_dict[key] = dictionary[key]
            # Replace the old dictionary with the updated dictionary
            dictionary.clear()
            dictionary.update(updated_dict)

        # Update tab names and any other necessary UI elements
        self.update_tab_name()
        self.update_active_tab_name()
        self.update_menu_action_text(closed_tab_index)

    # Define the slot function to update the slider
    def update_slider_from_entry(self):
        try:
            # Get the value from the index entry box
            value = int(self.index_entry.text())

            # Get the maximum value allowed for the slider
            max_value = self.index_slider.maximum()

            # Check if the entered value exceeds the maximum limit
            if value > max_value:
                # Set the value to the maximum limit
                value = max_value

                # Update the text in the entry box
                self.index_entry.setText(str(value))

            # Set the slider value
            self.index_slider.setValue(value)
        except ValueError:
            pass  # Handle the case where the text in the entry box is not a valid integer

    def connect_slider_to_index_dimension(self):
        # Get the value of the index dimension from the index_dim_entry
        index_dim_value = int(self.index_dim_entry.text())

        # Get the shape of the tensor data
        tensor_shape = self.tensor_data.shape

        # Get the maximum value for the slider from the shape of the tensor data
        max_slider_value = tensor_shape[index_dim_value] - 1

        # Set the maximum value of the slider
        self.index_slider.setMaximum(max_slider_value)
        self.index_slider.setMinimum(0)

        def update_index_entry(value):
            self.index_entry.setText(str(value))

        # Connect the slider's valueChanged signal to update the index_entry
        self.index_slider.valueChanged.connect(update_index_entry)

    def set_time_slice(self):
        if self.time_slice_button.isChecked():
            self.index_label.show()
            self.index_entry.show()
            self.index_slider.show()
            self.dim1_entry.setText("1")
            self.dim2_entry.setText("2")
            self.index_dim_entry.setText("0")
            self.cross_line_button.setChecked(False)
            self.inline_button.setChecked(False)
            self.connect_slider_to_index_dimension()
        else:
            self.dim1_entry.clear()
            self.dim2_entry.clear()
            self.index_dim_entry.clear()

    def set_cross_line(self):
        if self.cross_line_button.isChecked():
            self.index_label.show()
            self.index_entry.show()
            self.index_slider.show()
            self.dim1_entry.setText("1")
            self.dim2_entry.setText("0")
            self.index_dim_entry.setText("2")
            self.time_slice_button.setChecked(False)
            self.inline_button.setChecked(False)
            self.connect_slider_to_index_dimension()
        else:
            self.dim1_entry.clear()
            self.dim2_entry.clear()
            self.index_dim_entry.clear()

    def set_inline(self):
        if self.inline_button.isChecked():
            self.index_label.show()
            self.index_entry.show()
            self.index_slider.show()
            self.dim1_entry.setText("2")
            self.dim2_entry.setText("0")
            self.index_dim_entry.setText("1")
            self.time_slice_button.setChecked(False)
            self.cross_line_button.setChecked(False)
            self.connect_slider_to_index_dimension()
        else:
            self.dim1_entry.clear()
            self.dim2_entry.clear()
            self.index_dim_entry.clear()

    def load_tensor(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NumPy Tensor File", "", "NumPy Files (*.npy)")
        # Check if the file dialog was canceled and no file was selected
        if not file_path:
            print("No file selected, operation canceled.")
            return  # Exit the method without changing anything
        else:
            self.three_d()
            self.file_entry.setText(file_path)
            self.index_entry.setText("0")
            # Add the loaded file path to the dictionary
            file_name = file_path.split("/")[-1]  # Extract the file name from the path
            self.file_name = file_name.split('_', 1)[0]
            self.loaded_file_paths[self.file_name] = file_path
            self.file_path = file_path
            self.update_recent_files_submenu()  # Update the recent files submenu
            try:
                QApplication.setOverrideCursor(self.custom_cursor)
                print(f"Attempting to load tensor from: {file_path}")
                self.progress_bar.show()
                # Simulate data loading progress (replace with your actual loading logic)
                total_bytes = os.path.getsize(file_path)
                loaded_bytes = 0

                with open(file_path, "rb") as file:
                    chunk_size = 512000000  # Adjust as needed
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        loaded_bytes += len(chunk)
                        progress = int(loaded_bytes / total_bytes * 100)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()  # Update the UI

                self.tensor_data = np.load(file_path)
                self.add_tensor(self.file_name, self.tensor_data)
                print(f"Successfully loaded tensor. Shape: {self.tensor_data.shape}")
                self.show_tensor_info()
                self.progress_bar.hide()
                self.channel_index_entry.setText("0")
                if self.tensor_data.shape[-1] > 1:
                    self.set_slider_range(self.tensor_data.shape[-1])
                    self.channel_index_label.show()
                    self.channel_index_entry.show()
                    self.Channel_slider.show()
                else:
                    self.channel_index_label.hide()
                    self.channel_index_entry.hide()
                    self.Channel_slider.hide()
                # Check if the loaded tensor has more than four dimension
                if len(self.tensor_data.shape) > 4:
                    self.channel_label.show()
                    self.channel_entry.show()
                else:
                    self.channel_label.hide()
                    self.channel_entry.hide()
                sampling_interval = self.get_sampling_interval_from_file_name()
                current_index = self.tab_widget.currentIndex()
                current_widget = self.tab_widget.widget(current_index)
                if current_widget.layout():
                    canvas = current_widget.layout().itemAt(0).widget()

                    if isinstance(canvas, PyQtGraphCanvas):
                        # If the PyQtGraphCanvas is shown, hide the sampling interval UI elements
                        self.sampling_interval_label.hide()
                        self.sampling_interval_entry.hide()
                    elif isinstance(canvas, RoundedCanvas) and sampling_interval is None:
                        # If the RoundedCanvas is shown and there is no sampling interval, show the UI elements
                        self.sampling_interval_label.show()
                        self.sampling_interval_entry.show()

                if not sampling_interval is None:
                    self.sampling_interval_label.hide()
                    self.sampling_interval_entry.hide()

                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def plot_data3d(self):
        if self.tensor_data is not None:
            if self.canvas is not None:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Get the index of the currently active tab
                active_tab_index = self.tab_widget.currentIndex()

                # Retrieve the widget of the currently active tab
                active_tab_widget = self.tab_widget.widget(active_tab_index)

                # Assuming the canvas is the first widget in the layout of the active tab
                canvas_layout = active_tab_widget.layout()
                canvas = canvas_layout.itemAt(0).widget()

                # Get the state for the current tab
                is_3D_displayed = self.tab_3D_state.get(active_tab_index, False)

                # Retrieve the file name for the current data for the active tab
                current_data_name = self.tab_last_plotted_data_name.get(active_tab_index)

                # Compare the file names instead of the full file paths
                if current_data_name != self.file_name:
                    self.tab_volume_items.pop(active_tab_index, None)
                    # Clear the 3D plot from the canvas
                    while canvas.plot_widget.items:
                        item = canvas.plot_widget.items[0]
                        canvas.plot_widget.removeItem(item)
                        del item
                    self.tab_3D_state[active_tab_index] = False  # Update the state in the dictionary

                    # Update the dictionary with the new file name
                    self.tab_last_plotted_data_name[active_tab_index] = self.file_name

                # Check if a 3D plot is currently displayed and a 2D slice is to be plotted
                if is_3D_displayed and any([self.time_slice_button.isChecked(), self.cross_line_button.isChecked(),
                                            self.inline_button.isChecked()]):
                    self.tab_volume_items.pop(active_tab_index, None)
                    # Clear the 3D plot from the canvas
                    while canvas.plot_widget.items:
                        item = canvas.plot_widget.items[0]
                        canvas.plot_widget.removeItem(item)
                        del item
                    self.tab_3D_state[active_tab_index] = False  # Update the state in the dictionary

                # If a full 3D plot is created, update the state in the dictionary
                if not any([self.time_slice_button.isChecked(), self.cross_line_button.isChecked(),
                            self.inline_button.isChecked()]):
                    self.tab_3D_state[active_tab_index] = True
                    self.tab_volume_items.pop(active_tab_index, None)
                    # Clear all items from the plot widget if 3D button is checked
                    while canvas.plot_widget.items:
                        item = canvas.plot_widget.items[0]
                        canvas.plot_widget.removeItem(item)
                        del item

                data = np.copy(self.tensor_data)

                if data.shape[-1] > 1:
                    channel_index = int(self.channel_index_entry.text())
                    data = data[:, :, :, channel_index:channel_index + 1]

                full_data = data

                # Determine which slice to plot based on the button states
                if any([self.time_slice_button.isChecked(), self.cross_line_button.isChecked(),
                        self.inline_button.isChecked()]):
                    index = int(self.index_entry.text())
                    if self.time_slice_button.isChecked():
                        # Plotting the 2D slice along the 'time' axis
                        data = data[index, :, :, np.newaxis]
                    elif self.cross_line_button.isChecked():
                        # Plotting the 2D slice along the 'cross line' axis
                        data = data[:, :, index, np.newaxis]
                    elif self.inline_button.isChecked():
                        # Plotting the 2D slice along the 'inline' axis
                        data = data[:, index, :, np.newaxis]

                print('data shape', data.shape)
                # Assume the tensor_data shape is (depth, height, width, channel)
                depth, height, width, _ = data.shape

                # Extract intensity values from the tensor data
                intensity = data[..., 0]

                # Normalize the intensity values using z-score normalization
                mean_intensity = np.mean(intensity)
                std_intensity = np.std(intensity)
                intensity_z_score_normalized = (intensity - mean_intensity) / std_intensity
                intensity_z_score_normalized = np.clip(intensity_z_score_normalized, -3, 3)

                # Function to check if the base part of the string matches the given patterns
                def check_base_part(color_map, patterns):
                    # Remove the number at the end if it exists
                    base_part = re.sub(r'(_\d+)$', '', color_map)
                    # Check if the base part matches any of the patterns
                    return base_part in patterns

                norm_patterns = ['kmeans_clustered_amplitude']
                # Check if the base part of self.color_mapp matches the patterns
                if check_base_part(self.color_mapp, norm_patterns):
                    intensity_z_score_normalized = intensity

                facies_patterns = ['facies']
                # Check if the base part of self.color_mapp matches the patterns
                if check_base_part(self.color_mapp, facies_patterns):
                    intensity[intensity > 0.5] = 1
                    intensity[intensity < 0.5] = 0
                    intensity_z_score_normalized = intensity

                # Define the custom color schemes
                custom_color_mappings = {
                    'seismic_default': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000', '#FFFF00'],
                    'app._polarity': ['#000080', '#FFFFFF', '#BF0000'],
                    'inst._frequency': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                    'cos_phase': ['black', '#FFFFFF'],
                    'inst._phase': ['#FF69B4', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                    'envelope': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000', '#FFFF00'],
                    'inst._bandwidth': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                    'dominant_frequency': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                    'sweetness': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000', '#FFFF00'],
                    'rms_amplitude': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000', '#FFFF00'],
                    'faults': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000', '#FFFF00'],
                    'thresholded_amplitude': ['#A1FFFF', '#000080', '#4D4D4D', '#FFFFFF', '#614500', '#BF0000',
                                              '#FFFF00'],
                }

                def generate_hex_color_list(num_colors):
                    # Start with white
                    colors = ['#FFFFFF', '#25304D']

                    # Generate a color map using a matplotlib colormap
                    cmap = plt.get_cmap('jet', num_colors)

                    # Skip the first color (white) and generate the rest
                    for i in range(1, num_colors):
                        rgb = cmap(i)[:3]  # Get RGB from RGBA
                        colors.append(matplotlib.colors.rgb2hex(rgb))

                    return colors

                # Check the name of the reference and choose the color mapping
                if self.color_mapp in custom_color_mappings:
                    colors = custom_color_mappings[self.color_mapp]
                else:
                    # Get the unique values in the array, excluding zero
                    num_clusters = len(np.unique(intensity_z_score_normalized))
                    colors = generate_hex_color_list(num_clusters + 1)

                # Convert hexadecimal colors to RGB tuples and normalize to [0, 1] range
                def hex_to_rgb_normalized(hex_color):
                    # Initialize alpha to 1.0 for full opacity by default
                    alpha = 1.0

                    # Check if the color mapping is set to 'faults'
                    if self.color_mapp == 'faults':
                        # Check if the color is yellow
                        if hex_color == '#FFFF00':
                            alpha = 1.0  # Set alpha for yellow to 1 (fully opaque)
                        else:
                            alpha = 0.0  # Set alpha for other colors to 0 (fully transparent)

                    # Handle special cases for 'black' and 'white'
                    if hex_color == '#FFFFFF' or hex_color == 'white':
                        # Patterns to match
                        patterns = ['thresholded_amplitude', 'kmeans_clustered_amplitude', 'facies']
                        # Check if the base part of self.color_mapp matches the patterns
                        if check_base_part(self.color_mapp, patterns):
                            return 1.0, 1.0, 1.0, 0
                        else:
                            return 1.0, 1.0, 1.0, alpha
                    elif hex_color == 'black':
                        return 0.0, 0.0, 0.0, alpha

                    # Remove '#' if present
                    hex_color = hex_color.lstrip('#')

                    # Convert hex to RGB and add the alpha value
                    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)

                kmeans_patterns = ['kmeans_clustered_amplitude', 'facies']
                # Check if the base part of self.color_mapp matches the patterns
                if check_base_part(self.color_mapp, kmeans_patterns):
                    # Apply the conversion to all colors
                    color_palette = [hex_to_rgb_normalized(color) for color in colors]

                    # Assuming 'cluster_data' is your array of cluster indices
                    # Initialize an empty array for the mapped colors
                    colors_mapped = np.zeros((*intensity_z_score_normalized.shape, 4), dtype=np.float32)

                    # Map the value 0 to white
                    colors_mapped[intensity_z_score_normalized == 0] = color_palette[0]

                    # Get the unique values in the array, excluding zero
                    unique_values = np.unique(intensity_z_score_normalized)
                    unique_values = unique_values[unique_values != 0]  # Remove zero if present

                    # Map other values to their respective colors
                    for idx, value in enumerate(unique_values):
                        # Map each unique value to the corresponding color in the palette
                        # The index 'idx + 1' is used because 'color_palette[0]' is for zero
                        colors_mapped[intensity_z_score_normalized == value] = color_palette[idx + 1]

                else:
                    # Apply the conversion to all colors
                    gradient = np.array([hex_to_rgb_normalized(color) for color in colors], dtype=np.float32)

                    data_max = np.max(abs(intensity_z_score_normalized))
                    data_sym = - data_max
                    data_asym = np.min(intensity_z_score_normalized)

                    # Define the data type
                    custom_data_type = {
                        'seismic_default': data_sym,
                        'app._polarity': data_sym,
                        'inst._frequency': data_asym,
                        'cos_phase': data_sym,
                        'inst._phase': data_sym,
                        'envelope': data_asym,
                        'inst._bandwidth': data_asym,
                        'dominant_frequency': data_asym,
                        'sweetness': data_asym,
                        'rms_amplitude': data_asym,
                        'faults': data_sym,
                        'swin_fault': data_sym,
                        'thresholded_amplitude': data_sym,
                    }

                    # Check the name of the reference and choose the custom_data_type
                    if self.color_mapp in custom_data_type:
                        data_min = custom_data_type[self.color_mapp]

                    # Create an interpolation function based on the gradient
                    interpolation_function = interp1d(np.linspace(data_min, data_max, num=len(gradient)), gradient,
                                                      axis=0)

                    # Use the interpolation function to map the normalized intensities to colors
                    colors_mapped = interpolation_function(intensity_z_score_normalized)

                # Reshape the colors_mapped array to match the shape of the intensity data
                colors_mapped_reshaped = colors_mapped.reshape(depth, height, width, 4)

                # Create a 4D array (x, y, z, RGBA) with dtype=ubyte for GLVolumeItem
                volume_data = np.empty((depth, height, width, 4), dtype=np.ubyte)

                colors_mapped_reshaped[..., 3] = np.round(colors_mapped_reshaped[..., 3])

                # Populate the RGB channels
                volume_data[..., :4] = (colors_mapped_reshaped * 255).astype(np.ubyte)

                # Create a GLVolumeItem for rendering
                volume_item = GLVolumeItem(volume_data, sliceDensity=1, smooth=True)

                # Calculate the global center of the data
                global_center_x = (full_data.shape[0] - 1) / 2.0
                global_center_y = (full_data.shape[1] - 1) / 2.0
                global_center_z = (full_data.shape[2] - 1) / 2.0

                # Apply the offset when translating the volume item
                volume_item.translate(-global_center_x, -global_center_y,
                                      -global_center_z)

                # Retrieve the index for the current slice
                slice_index = int(self.index_entry.text())

                # Calculate the position offset for the current slice based on the global center
                if self.inline_button.isChecked():
                    volume_item.translate(+global_center_x, +global_center_y - (global_center_z - global_center_y),
                                          -slice_index + 2 * global_center_y + (global_center_z - global_center_y))

                if self.time_slice_button.isChecked():
                    volume_item.translate(+global_center_x - (global_center_y - global_center_x),
                                          + global_center_y - (global_center_z - global_center_y),
                                          + slice_index + (global_center_z - global_center_x))

                if self.cross_line_button.isChecked():
                    volume_item.translate(+global_center_x, +global_center_y,
                                          + slice_index)

                if self.cross_line_button.isChecked() or self.inline_button.isChecked() or self.time_slice_button.isChecked():
                    # Calculate the global center of the data
                    global_center_x = (full_data.shape[0] - 1) / 2.0
                    global_center_y = (full_data.shape[1] - 1) / 2.0
                    global_center_z = (full_data.shape[2] - 1) / 2.0

                    # Apply the offset when translating the volume item
                    volume_item.translate(-global_center_x, -global_center_y, 0)

                # Rotate around the Y-axis by 90 degrees
                volume_item.rotate(90, 0, 1, 0)

                # Additional rotations based on the selected button
                if self.cross_line_button.isChecked():
                    # Rotate around the Z-axis by 90 degrees
                    volume_item.rotate(90, 0, 0, 1)

                if self.time_slice_button.isChecked():
                    # Rotate around the Y-axis by -90 degrees
                    volume_item.rotate(90, 0, 1, 0)

                # Add the volume item to the plot widget
                canvas.plot_widget.addItem(volume_item)
                if active_tab_index not in self.tab_volume_items:
                    self.tab_volume_items[active_tab_index] = []
                self.tab_volume_items[active_tab_index].append(volume_item)
                self.print_tab_volumes()

                num_squares = 10

                grid_spacing_x = (full_data.shape[2] / num_squares)
                grid_spacing_y = (full_data.shape[1] / num_squares)
                grid_spacing_z = (full_data.shape[0] / num_squares)

                # Check if the grid for the current tab exists in the grids dictionary
                if active_tab_index not in self.grids:
                    # Create a new grid instance for the current tab
                    self.grids[active_tab_index] = GLGridItem()
                    self.grids[active_tab_index].setSize(x=2 * full_data.shape[2],
                                                         y=2 * full_data.shape[1],
                                                         z=2 * full_data.shape[
                                                             0])  # Customize the size if needed
                    self.grids[active_tab_index].setSpacing(x=grid_spacing_x, y=grid_spacing_y, z=grid_spacing_z)
                    canvas.plot_widget.addItem(self.grids[active_tab_index])  # Add the new grid to the GLViewWidget
                    # Calculate the range of your data for labeling
                    # Extract the file name from the file path
                    filename = os.path.basename(self.file_path)

                    # Extract the min and max values from the filename
                    parts = filename.split('_tensor_')[-1].split('_')
                    x_min, x_max, y_min, y_max = map(float,
                                                     parts[:4])  # Only take the first four values after 'tensor_'

                    print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

                    # Determine the size of the grid based on the maximum dimension
                    grid_size_x = 2 * full_data.shape[1]
                    grid_size_y = 2 * full_data.shape[2]
                    # Calculate the positions for the labels based on the grid size
                    label_positions_x = np.linspace(-grid_size_y / 4, grid_size_y / 4, 5 + 1)
                    label_positions_y = np.linspace(-grid_size_x / 4, grid_size_x / 4, 5 + 1)

                    # Function to format the label text with two decimal places
                    def format_label(value):
                        return f"{int(value)}"

                    # Create a QFont object with the desired font size
                    font = QFont()
                    font.setPointSize(10)  # Set the font size to 10 points

                    # Ensure the dictionaries are initialized
                    if not hasattr(self, 'grid_labels'):
                        self.grid_labels = {}
                    if not hasattr(self, 'grid_axis_labels'):
                        self.grid_axis_labels = {}

                    # Ensure there is a list to hold the labels for the active tab
                    self.grid_labels.setdefault(active_tab_index, [])
                    self.grid_axis_labels.setdefault(active_tab_index, [])

                    # Map global coordinate values to local grid positions and add labels
                    for i, (local_x, local_y) in enumerate(zip(label_positions_x, label_positions_y)):
                        global_x = np.interp(local_x, [-grid_size_x / 8, grid_size_x / 8], [x_min, x_max])
                        global_y = np.interp(local_y, [-grid_size_y / 8, grid_size_y / 8], [y_min, y_max])

                        # Add labels along the X-axis edge, except at the corners
                        x_label_bottom = GLTextItem(pos=(local_x, -grid_size_x / 2, 0), text=format_label(global_x),
                                                    font=font)
                        canvas.plot_widget.addItem(x_label_bottom)
                        self.grid_labels[active_tab_index].append(x_label_bottom)

                        x_label_top = GLTextItem(pos=(local_x, grid_size_x / 2, 0), text=format_label(global_x),
                                                 font=font)
                        canvas.plot_widget.addItem(x_label_top)
                        self.grid_labels[active_tab_index].append(x_label_top)

                        # Add labels along the Y-axis edge, except at the corners
                        y_label_left = GLTextItem(pos=(-grid_size_y / 2, local_y, 0), text=format_label(global_y),
                                                  font=font)
                        canvas.plot_widget.addItem(y_label_left)
                        self.grid_labels[active_tab_index].append(y_label_left)

                        y_label_right = GLTextItem(pos=(grid_size_y / 2, local_y, 0), text=format_label(global_y),
                                                   font=font)
                        canvas.plot_widget.addItem(y_label_right)
                        self.grid_labels[active_tab_index].append(y_label_right)

                    # Create a QFont object with the desired font size for the axis labels
                    axis_font = QFont()
                    axis_font.setPointSize(14)  # Set the font size to 14 points for the axis labels

                    # Calculate the middle position for the axis labels
                    middle_x = (label_positions_x[0] + label_positions_x[-1]) / 2
                    middle_y = (label_positions_y[0] + label_positions_y[-1]) / 2
                    offset_x = 0.1 * grid_size_x  # Increased offset
                    offset_y = 0.1 * grid_size_y

                    # Add X-Axis labels with increased offset and specified font size
                    x_axis_label_bottom = GLTextItem(pos=(middle_x, -grid_size_x / 2 - offset_x, 0), text="X-Axis",
                                                     font=axis_font)
                    canvas.plot_widget.addItem(x_axis_label_bottom)
                    self.grid_axis_labels[active_tab_index].append(x_axis_label_bottom)

                    x_axis_label_top = GLTextItem(pos=(middle_x, grid_size_x / 2 + offset_x, 0), text="X-Axis",
                                                  font=axis_font)
                    canvas.plot_widget.addItem(x_axis_label_top)
                    self.grid_axis_labels[active_tab_index].append(x_axis_label_top)

                    # Add Y-Axis labels with increased offset and specified font size
                    y_axis_label_left = GLTextItem(pos=(-grid_size_y / 2 - offset_y, middle_y, 0), text="Y-Axis",
                                                   font=axis_font)
                    canvas.plot_widget.addItem(y_axis_label_left)
                    self.grid_axis_labels[active_tab_index].append(y_axis_label_left)

                    y_axis_label_right = GLTextItem(pos=(grid_size_y / 2 + offset_y, middle_y, 0), text="Y-Axis",
                                                    font=axis_font)
                    canvas.plot_widget.addItem(y_axis_label_right)
                    self.grid_axis_labels[active_tab_index].append(y_axis_label_right)

                    if active_tab_index in self.grid_color and self.grid_color[active_tab_index]:
                        self.set_grid_color()
                    # Additional rotations based on the selected button
                    if not self.Three_D_button.isChecked():
                        # Rotate around the Z-axis by 90 degrees
                        self.grids[active_tab_index].rotate(90, 0, 0, 1)
                        for item in self.grid_axis_labels[active_tab_index]:
                            item.rotate(90, 0, 0, 1)
                        for item in self.grid_labels[active_tab_index]:
                            item.rotate(90, 0, 0, 1)

                else:
                    # The grid already exists for the current tab, so retrieve it
                    grid = self.grids[active_tab_index]
                    grid_labels = self.grid_labels[active_tab_index]
                    grid_axis_labels = self.grid_axis_labels[active_tab_index]
                    # Store the visibility state of the current grid
                    grid_visibility = grid.visible()
                    # Check if the grid is in the list before removing it
                    if grid in canvas.plot_widget.items:
                        canvas.plot_widget.removeItem(grid)

                    # Remove items from grid_labels if they are present
                    for label in grid_labels:
                        if label in canvas.plot_widget.items:
                            canvas.plot_widget.removeItem(label)

                    # Remove items from grid_axis_labels if they are present
                    for axis_label in grid_axis_labels:
                        if axis_label in canvas.plot_widget.items:
                            canvas.plot_widget.removeItem(axis_label)

                    # Add the grid back to ensure it's visible
                    # Create a new grid instance for the current tab
                    self.grids[active_tab_index] = GLGridItem()
                    self.grids[active_tab_index].setSize(x=2 * full_data.shape[2],
                                                         y=2 * full_data.shape[1],
                                                         z=2 * full_data.shape[
                                                             0])  # Customize the size if needed
                    # Set the visibility of the new grid based on the previous grid's visibility
                    self.grids[active_tab_index].setSpacing(x=grid_spacing_x, y=grid_spacing_y, z=grid_spacing_z)
                    self.grids[active_tab_index].setVisible(grid_visibility)
                    canvas.plot_widget.addItem(self.grids[active_tab_index])  # Add the new grid to the GLViewWidget
                    # Calculate the range of your data for labeling
                    # Extract the file name from the file path
                    filename = os.path.basename(self.file_path)

                    # Extract the min and max values from the filename
                    parts = filename.split('_tensor_')[-1].split('_')
                    x_min, x_max, y_min, y_max = map(float,
                                                     parts[:4])  # Only take the first four values after 'tensor_'

                    print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

                    # Determine the size of the grid based on the maximum dimension
                    grid_size_x = 2 * full_data.shape[1]
                    grid_size_y = 2 * full_data.shape[2]
                    # Calculate the positions for the labels based on the grid size
                    label_positions_x = np.linspace(-grid_size_y / 4, grid_size_y / 4, 5 + 1)
                    label_positions_y = np.linspace(-grid_size_x / 4, grid_size_x / 4, 5 + 1)

                    # Function to format the label text with two decimal places
                    def format_label(value):
                        return f"{int(value)}"

                    # Create a QFont object with the desired font size
                    font = QFont()
                    font.setPointSize(10)  # Set the font size to 10 points

                    # Ensure the dictionaries are initialized
                    if not hasattr(self, 'grid_labels'):
                        self.grid_labels = {}
                    if not hasattr(self, 'grid_axis_labels'):
                        self.grid_axis_labels = {}

                    # Ensure there is a list to hold the labels for the active tab
                    self.grid_labels.setdefault(active_tab_index, [])
                    self.grid_axis_labels.setdefault(active_tab_index, [])

                    # Map global coordinate values to local grid positions and add labels
                    for i, (local_x, local_y) in enumerate(zip(label_positions_x, label_positions_y)):
                        global_x = np.interp(local_x, [-grid_size_x / 8, grid_size_x / 8], [x_min, x_max])
                        global_y = np.interp(local_y, [-grid_size_y / 8, grid_size_y / 8], [y_min, y_max])

                        # Add labels along the X-axis edge, except at the corners
                        x_label_bottom = GLTextItem(pos=(local_x, -grid_size_x / 2, 0), text=format_label(global_x),
                                                    font=font)
                        canvas.plot_widget.addItem(x_label_bottom)
                        self.grid_labels[active_tab_index].append(x_label_bottom)

                        x_label_top = GLTextItem(pos=(local_x, grid_size_x / 2, 0), text=format_label(global_x),
                                                 font=font)
                        canvas.plot_widget.addItem(x_label_top)
                        self.grid_labels[active_tab_index].append(x_label_top)

                        # Add labels along the Y-axis edge, except at the corners
                        y_label_left = GLTextItem(pos=(-grid_size_y / 2, local_y, 0), text=format_label(global_y),
                                                  font=font)
                        canvas.plot_widget.addItem(y_label_left)
                        self.grid_labels[active_tab_index].append(y_label_left)

                        y_label_right = GLTextItem(pos=(grid_size_y / 2, local_y, 0), text=format_label(global_y),
                                                   font=font)
                        canvas.plot_widget.addItem(y_label_right)
                        self.grid_labels[active_tab_index].append(y_label_right)

                    # Create a QFont object with the desired font size for the axis labels
                    axis_font = QFont()
                    axis_font.setPointSize(14)  # Set the font size to 14 points for the axis labels

                    # Calculate the middle position for the axis labels
                    middle_x = (label_positions_x[0] + label_positions_x[-1]) / 2
                    middle_y = (label_positions_y[0] + label_positions_y[-1]) / 2
                    offset_x = 0.1 * grid_size_x  # Increased offset
                    offset_y = 0.1 * grid_size_y

                    # Add X-Axis labels with increased offset and specified font size
                    x_axis_label_bottom = GLTextItem(pos=(middle_x, -grid_size_x / 2 - offset_x, 0), text="X-Axis",
                                                     font=axis_font)
                    canvas.plot_widget.addItem(x_axis_label_bottom)
                    self.grid_axis_labels[active_tab_index].append(x_axis_label_bottom)

                    x_axis_label_top = GLTextItem(pos=(middle_x, grid_size_x / 2 + offset_x, 0), text="X-Axis",
                                                  font=axis_font)
                    canvas.plot_widget.addItem(x_axis_label_top)
                    self.grid_axis_labels[active_tab_index].append(x_axis_label_top)

                    # Add Y-Axis labels with increased offset and specified font size
                    y_axis_label_left = GLTextItem(pos=(-grid_size_y / 2 - offset_y, middle_y, 0), text="Y-Axis",
                                                   font=axis_font)
                    canvas.plot_widget.addItem(y_axis_label_left)
                    self.grid_axis_labels[active_tab_index].append(y_axis_label_left)

                    y_axis_label_right = GLTextItem(pos=(grid_size_y / 2 + offset_y, middle_y, 0), text="Y-Axis",
                                                    font=axis_font)
                    canvas.plot_widget.addItem(y_axis_label_right)
                    self.grid_axis_labels[active_tab_index].append(y_axis_label_right)

                    # and set their visibility
                    for label in self.grid_labels[active_tab_index]:
                        label.setVisible(grid_visibility)

                    for axis_label in self.grid_axis_labels[active_tab_index]:
                        axis_label.setVisible(grid_visibility)

                    if active_tab_index in self.grid_color and self.grid_color[active_tab_index]:
                        self.set_grid_color()

                    if not self.Three_D_button.isChecked():
                        # Rotate around the Z-axis by 90 degrees
                        self.grids[active_tab_index].rotate(90, 0, 0, 1)
                        for item in self.grid_axis_labels[active_tab_index]:
                            item.rotate(90, 0, 0, 1)
                        for item in self.grid_labels[active_tab_index]:
                            item.rotate(90, 0, 0, 1)

                # Use the active_tab_index to access the items from the dictionaries
                grid_item = self.grids_initial.get(active_tab_index)
                x_axis_label_bottom_item = self.x_axis_label_bottom_initial.get(active_tab_index)
                x_axis_label_top_item = self.x_axis_label_top_initial.get(active_tab_index)
                y_axis_label_left_item = self.y_axis_label_left_initial.get(active_tab_index)
                y_axis_label_right_item = self.y_axis_label_right_initial.get(active_tab_index)

                active_tab_index = self.tab_widget.currentIndex()
                grid = self.grids[active_tab_index]
                grid_x_axis_label_bottom = self.x_axis_label_bottom_initial[active_tab_index]
                grid_x_axis_label_top = self.x_axis_label_top_initial[active_tab_index]
                grid_y_axis_label_left = self.y_axis_label_left_initial[active_tab_index]
                grid_y_axis_label_right = self.y_axis_label_right_initial[active_tab_index]
                if grid_item is not None:
                    if not grid_item.visible():
                        grid.hide()
                        grid_x_axis_label_bottom.hide()
                        grid_x_axis_label_top.hide()
                        grid_y_axis_label_left.hide()
                        grid_y_axis_label_right.hide()
                        # Hide all items in the list associated with the active_tab_index in grid_labels
                        for item in self.grid_labels[active_tab_index]:
                            item.hide()
                        # Hide all items in the list associated with the active_tab_index in grid_axis_labels
                        for item in self.grid_axis_labels[active_tab_index]:
                            item.hide()

                        self.grids_initial[active_tab_index] = None

                # Check if the items are in the canvas before removing them
                if grid_item and grid_item in canvas.plot_widget.items:
                    canvas.plot_widget.removeItem(grid_item)
                    canvas.plot_widget.removeItem(x_axis_label_bottom_item)
                    canvas.plot_widget.removeItem(x_axis_label_top_item)
                    canvas.plot_widget.removeItem(y_axis_label_left_item)
                    canvas.plot_widget.removeItem(y_axis_label_right_item)

                canvas.plot_widget.opts['distance'] = 3.5 * max(volume_data.shape)  # Set the camera distance
                # Update the plot widget
                canvas.plot_widget.update()
                QApplication.restoreOverrideCursor()

    def print_tab_volumes(self):
        for tab_index, volumes in self.tab_volume_items.items():
            print(f"Tab {tab_index + 1} volumes:")
            for i, volume_item in enumerate(volumes, start=1):
                print(f"  Volume {i}")

    def show_tensor_info(self):
        # Display tensor shape and other info
        self.shape_label.setText(f"Tensor Shape: {self.tensor_data.shape}")

    def get_sampling_interval_from_file_name(self):
        if self.file_name is not None:
            file_name = os.path.basename(self.loaded_file_paths[self.file_name.split('_')[0]])
            print("file_name:", file_name)
            pattern = r"_sampling_interval(\d+\.\d+)ms"
            match = re.search(pattern, file_name)
            print("match:", match)  # Print the match object
            if match:
                return float(match.group(1))
            else:
                return 4

    def plot_tensor(self):
        if self.tensor_data is not None:
            try:
                QApplication.setOverrideCursor(self.custom_cursor)
                # Get the index of the currently active tab
                active_tab_index = self.tab_widget.currentIndex()

                # Retrieve the widget of the currently active tab
                active_tab_widget = self.tab_widget.widget(active_tab_index)

                # Assuming the canvas is the first widget in the layout of the active tab
                canvas_layout = active_tab_widget.layout()
                canvas = canvas_layout.itemAt(0).widget()

                # Retrieve selected dimensions and channel
                dim1 = int(self.dim1_entry.text())
                dim2 = int(self.dim2_entry.text())
                index_dim = int(self.index_dim_entry.text())
                if len(self.tensor_data.shape) > 4:
                    channel_dim = int(self.channel_entry.text())
                else:
                    channel_dim = int(3)

                if self.tensor_data.shape[-1] > 1:
                    channel_index = int(self.channel_index_entry.text())
                else:
                    channel_index = 0

                # Retrieve selected index
                selected_index = int(self.index_entry.text())

                # Extract the 2D slice from the tensor
                if channel_dim < len(self.tensor_data.shape):
                    slice_2d = np.squeeze(
                        np.take(self.tensor_data, selected_index, axis=index_dim)[..., channel_index])

                    # Determine title and file name based on conditions
                    if int(self.index_dim_entry.text()) < int(self.dim1_entry.text()) and \
                            int(self.index_dim_entry.text()) < int(self.dim2_entry.text()) and \
                            int(self.index_dim_entry.text()) < int(channel_dim):
                        title_word = "Time Slice"
                    else:
                        if int(self.index_dim_entry.text()) > int(self.dim1_entry.text()):
                            title_word = "Crossline"
                        else:
                            title_word = "Inline"

                    # Check if dimension 2 is smaller than other dimensions
                    if int(self.dim2_entry.text()) < int(self.dim1_entry.text()) and \
                            int(self.dim2_entry.text()) < int(self.index_dim_entry.text()) and \
                            int(self.dim2_entry.text()) < int(channel_dim):
                        flip_vertical_axis = True
                    else:
                        flip_vertical_axis = False

                    # Assuming tensor_data is your 4D tensor
                    tensor_max_abs = np.max(np.abs(slice_2d))
                    self.interpolated_slice_2d[active_tab_index] = ndimage.zoom(slice_2d, zoom=8,
                                                                                order=3)  # zoom=2 is an example, adjust as needed
                    tensor_min = np.min(slice_2d)
                    tensor_max = np.max(slice_2d)
                    # Print the minimum and maximum values within the slice for debugging
                    print("Min value in the slice:", tensor_min)
                    print("Max value in the slice:", tensor_max)
                    print("Max absolute value in the slice:", tensor_max_abs)
                    # Define the custom color schemes
                    custom_color_mappings = {
                        'seismic_default': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
                        'app._polarity': ['#000080', 'white', '#BF0000'],
                        'inst._frequency': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                        'cos_phase': ['black', 'white'],
                        'inst._phase': ['#FF69B4', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                        'envelope': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
                        'inst._bandwidth': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080', '#FF00FF'],
                        'dominant_frequency': ['black', '#BF0000', '#FFFF00', '#008000', '#00FFFF', '#000080',
                                               '#FF00FF'],
                        'sweetness': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
                        'rms_amplitude': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
                        'faults': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000', '#FFFF00'],
                        'thresholded_amplitude': ['#A1FFFF', '#000080', '#4D4D4D', 'white', '#614500', '#BF0000',
                                                  '#FFFF00'],
                    }

                    def generate_hex_color_list(num_colors):
                        # Start with white
                        colors = ['#FFFFFF', '#25304D']

                        # Generate a color map using a matplotlib colormap
                        cmap = plt.get_cmap('jet', num_colors)

                        # Skip the first color (white) and generate the rest
                        for i in range(1, num_colors):
                            rgb = cmap(i)[:3]  # Get RGB from RGBA
                            colors.append(matplotlib.colors.rgb2hex(rgb))

                        return colors

                    # Check the name of the reference and choose the color mapping
                    if self.color_mapp in custom_color_mappings:
                        colors = custom_color_mappings[self.color_mapp]
                    else:
                        # Get the unique values in the array, excluding zero
                        num_clusters = len(np.unique(slice_2d))
                        colors = generate_hex_color_list(num_clusters)

                    # Define the color bar limits
                    custom_color_bar_limits = {
                        'seismic_default': (-tensor_max_abs, tensor_max_abs),
                        'app._polarity': (-tensor_max_abs, tensor_max_abs),
                        'inst._frequency': (tensor_min, tensor_max),
                        'cos_phase': (-tensor_max_abs, tensor_max_abs),
                        'inst._phase': (-tensor_max_abs, tensor_max_abs),
                        'envelope': (tensor_min, tensor_max),
                        'inst._bandwidth': (tensor_min, tensor_max),
                        'dominant_frequency': (tensor_min, tensor_max),
                        'sweetness': (tensor_min, tensor_max),
                        'rms_amplitude': (tensor_min, tensor_max),
                        'faults': (-tensor_max_abs, tensor_max_abs),
                        'thresholded_amplitude': (-tensor_max_abs, tensor_max_abs),
                    }

                    # Check the name of the reference and choose the color bar limits
                    if self.color_mapp in custom_color_bar_limits:
                        color_bar_limits = custom_color_bar_limits[self.color_mapp]
                    else:
                        color_bar_limits = (tensor_min, tensor_max)

                    # Assuming tensor_max_abs, tensor_min, and tensor_max are defined elsewhere in the code
                    # Print a success message
                    print(f"Color bar limits for {self.color_mapp} are set to {color_bar_limits}.")

                    # Create a custom colormap using matplotlib
                    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

                    # Convert the custom colormap to a lookup table (LUT)
                    lut = custom_cmap(np.linspace(0, 1, 256))
                    lut = (lut * 255).astype(np.uint8)
                    # Clear the canvas before plotting a new image
                    canvas.plotItem.clear()
                    # Remove the existing color bar if it exists
                    if hasattr(self, 'cbar') and self.cbar.get(active_tab_index) is None:
                        # Create the HistogramLUTItem
                        self.cbar[active_tab_index] = CustomColorBarItem(orientation='left')
                    # Check if the vertical axis needs to be flipped
                    if flip_vertical_axis:
                        # Create the ImageItem with the slice_2d data
                        self.img_item[active_tab_index] = pg.ImageItem(self.interpolated_slice_2d[active_tab_index])
                        self.img_item[active_tab_index].setLookupTable(lut)  # Apply the LUT to the ImageItem
                        transform = pg.QtGui.QTransform()
                        transform.rotate(-90)
                        self.img_item[active_tab_index].setTransform(transform)
                        self.img_item[active_tab_index].setLevels(color_bar_limits)
                        canvas.plotItem.addItem(self.img_item[active_tab_index])  # Add the ImageItem to the canvas
                        # Access the PlotItem for further customization
                        plot_item = canvas.getPlotItem()
                        plot_item.setAspectLocked(True)
                        # Reset the zoom to fit the new image
                        plot_item.getViewBox().autoRange()
                        # Link the HistogramLUTItem to the ImageItem
                        self.cbar[active_tab_index].setImageItem(self.img_item[active_tab_index])

                        # Set the levels for the histogram
                        self.cbar[active_tab_index].setLevels(color_bar_limits)

                        # Calculate the number of colors
                        num_colors = len(colors)

                        # Create a list of positions for the color stops, ranging from 0 to 1
                        # The middle stop should be at 0.5, representing zero
                        stops = [(i / (num_colors - 1)) for i in range(num_colors)]

                        # Create the ColorMap object with symmetrical stops around the middle color
                        color_map = pg.ColorMap(stops, [pg.mkColor(c) for c in colors])
                        self.cbar[active_tab_index].setColorMap(color_map)

                        # Add the HistogramLUTItem to the layout to the right of the image
                        plot_item.layout.addItem(self.cbar[active_tab_index], 2, 4)
                        # Add x-axis title with a specific font size
                        if title_word == "Crossline":
                            xlabel = "Inline"
                        else:
                            xlabel = "Crossline"
                        # Move the x-axis to the top
                        plot_item.getAxis('bottom').setZValue(-10000)  # This moves the x-axis to the top
                        plot_item.getAxis('top').setZValue(10000)
                        plot_item.getAxis('top').setScale(0.125)
                        plot_item.showAxis('left')
                        plot_item.getAxis('left').setScale(0.125 * self.get_sampling_interval_from_file_name())
                        plot_item.showAxis('top')
                        plot_item.hideAxis('bottom')
                        plot_item.setLabel('top', xlabel)  # This sets the x-axis label to the top
                        plot_item.setLabel('left', 'TWT (ms)')  # This sets the x-axis label to the top
                        plot_item.setTitle(f"{title_word} {selected_index}", axis='bottom')
                    else:
                        # Create the ImageItem with the slice_2d data
                        self.img_item[active_tab_index] = pg.ImageItem(self.interpolated_slice_2d[active_tab_index])
                        self.img_item[active_tab_index].setLookupTable(lut)  # Apply the LUT to the ImageItem
                        self.img_item[active_tab_index].setLevels(color_bar_limits)
                        canvas.plotItem.addItem(self.img_item[active_tab_index])  # Add the ImageItem to the canvas
                        plot_item = canvas.getPlotItem()
                        plot_item.setAspectLocked(True)

                        # Reset the zoom to fit the new image
                        plot_item.getViewBox().autoRange()

                        # Link the HistogramLUTItem to the ImageItem
                        self.cbar[active_tab_index].setImageItem(self.img_item[active_tab_index])

                        # Set the levels for the histogram
                        self.cbar[active_tab_index].setLevels(color_bar_limits)

                        # Calculate the number of colors
                        num_colors = len(colors)

                        # Create a list of positions for the color stops, ranging from 0 to 1
                        # The middle stop should be at 0.5, representing zero
                        stops = [(i / (num_colors - 1)) for i in range(num_colors)]

                        # Create the ColorMap object with symmetrical stops around the middle color
                        color_map = pg.ColorMap(stops, [pg.mkColor(c) for c in colors])
                        self.cbar[active_tab_index].setColorMap(color_map)

                        # Add the HistogramLUTItem to the layout to the right of the image
                        plot_item.layout.addItem(self.cbar[active_tab_index], 2, 4)
                        # Retrieve the sampling interval value
                        sampling_interval = self.get_sampling_interval_from_file_name()
                        if sampling_interval is None:
                            print("None")
                            # Second condition: Use the user input (self.sampling_interval_entry.text())
                            sampling_interval = float(
                                self.sampling_interval_entry.text()) if self.sampling_interval_entry.text() else None
                        if sampling_interval is None:
                            # Third condition: If user input is empty, use 1.0
                            sampling_interval = 1.0
                        plot_item.showAxis('left')
                        plot_item.showAxis('bottom')
                        plot_item.getAxis('bottom').setScale(0.125)
                        plot_item.getAxis('left').setScale(0.125)
                        plot_item.hideAxis('top')
                        plot_item.setLabel('bottom', 'Crossline')  # This sets the x-axis label to the top
                        plot_item.setLabel('left', 'Inline')
                        plot_item.setTitle(f"{title_word}  {int(selected_index * sampling_interval)} ms", axis='bottom')

                    # Display the plot in the initial window
                    layout = self.centralWidget().layout()

                    # Check if there is a widget at index 1
                    if layout.count() > 1:
                        # Display the plot in the initial window
                        canvas = canvas_layout.itemAt(0).widget()
                        canvas.setFocusPolicy(Qt.NoFocus)
                        # Print the widget type for debugging
                        print(f"Widget type at index 1: {type(canvas)}")

                        # Check if the widget is a RoundedCanvas
                        if isinstance(canvas, RoundedCanvas):
                            # Use the existing canvas to show the plot
                            canvas.show()
                            canvas.plotItem.getViewBox().autoRange()

                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    # Set the global application icon
    app_icon = QIcon("icon.png")
    app.setWindowIcon(app_icon)

    try:
        window = TensorVisualizer()
        window.setGeometry(100, 100, 1200, 800)
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        QApplication.restoreOverrideCursor()
        import traceback

        traceback.print_exc()
        QMessageBox.critical("Error", f"An error occurred: {e}")

        sys.exit(1)
