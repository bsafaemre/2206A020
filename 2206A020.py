import sys
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMenu, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QListWidget)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                x = data.data
                y = data.target
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                x = data.data
                y = data.target
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                x = data.data
                y = data.target
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
                x = data.data
                y = data.target
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Split data
            split_option = self.split_combo.currentText()

            if split_option == "80-20 (Train-Test)":
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

            elif split_option == "70-15-15 (Train-Val-Test)":
                X_temp, self.X_test, y_temp, self.y_test = \
                    model_selection.train_test_split(x, y, test_size=0.15, random_state=42)
                self.X_train, self.X_val, self.y_train, self.y_val = \
                    model_selection.train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.15 / 0.85 ≈ 0.1765

            elif split_option == "60-20-20 (Train-Val-Test)":
                X_temp, self.X_test, y_temp, self.y_test = \
                    model_selection.train_test_split(x, y, test_size=0.20, random_state=42)
                self.X_train, self.X_val, self.y_train, self.y_val = \
                    model_selection.train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.20 / 0.80 = 0.25

            
            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    # missing data handling
                    handle_option = self.missing_combo.currentText()
                    if handle_option == "Mean Imputation":
                        from sklearn.impute import SimpleImputer
                        imputer=SimpleImputer(strategy="mean")
                        X=pd.DataFrame(imputer.fit_transform(X),columns=X.columns)
                        missing_msg="Mean Imputation applied."

                    elif handle_option == "Interpolation":
                        X = X.interpolate()
                        missing_msg="Interpolation applied."
                    
                    elif handle_option == "Forward Fill":
                        X = X.ffill()
                        missing_msg="Forward Fill applied."
                    
                    elif handle_option == "Backward Fill":
                        X = X.bfill()
                        missing_msg="Backward Fill applied."


                    # Simulate missing values
                    import random
                    missing_ratio = 0.05  # %5 eksik veri
                    X = pd.DataFrame(X)   # X numpy ise dönüştür
                    for _ in range(int(X.size * missing_ratio)):
                        i = random.randint(0, X.shape[0]-1)
                        j = random.randint(0, X.shape[1]-1)
                        X.iat[i, j] = np.nan

                    self.X_train, self.X_test, self.y_train, self.y_test = \
                    model_selection.train_test_split(X, y, test_size=test_size, random_state=42)

                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name} | {missing_msg}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        # missing data handling
        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "No Handling",
            "Mean Imputation",
            "Interpolation",
            "Forward Fill",
            "Backward Fill"
        ])

        
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        
        # Train/Val/Test split
        self.split_combo = QComboBox()
        self.split_combo.addItems([
            "80-20 (Train-Test)",
            "70-15-15 (Train-Val-Test)",
            "60-20-20 (Train-Val-Test)"
        ])
        data_layout.addWidget(QLabel("Split:"))
        data_layout.addWidget(self.split_combo)

        self.kfold_checkbox = QCheckBox("Use k-Fold CV")
        data_layout.addWidget(self.kfold_checkbox)


        data_layout.addWidget(QLabel("Missing Data:"))
        data_layout.addWidget(self.missing_combo)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

        # k-Fold 
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)

        data_layout.addWidget(QLabel("k-Fold:"))
        data_layout.addWidget(self.kfold_spin)

        # Plotly toggle
        self.use_plotly_checkbox = QCheckBox("Use Plotly for Visuals")
        data_layout.addWidget(self.use_plotly_checkbox)


    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Generative Adversarial Networks", self.create_gan_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QGridLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox",
             "loss_function" : ["MSE", "MAE","Huber"]}
        )
        regression_layout.addWidget(lr_group, 0, 0)


        # Support Vector Regression
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "epsilon": "double",
             "loss_function" : ["MSE", "MAE","Huber"]}
        )
        regression_layout.addWidget(svr_group, 0, 1)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"],
             "loss_function" : ["Cross-Entropy"]}
        )
        regression_layout.addWidget(logistic_group, 1, 0)

        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QGridLayout()

        
        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "priors": "lineedit"}
        )
        classification_layout.addWidget(nb_group, 0, 0)

        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int",
             "loss_function": ["Cross-Entropy", "Hinge"]}
        )
        classification_layout.addWidget(svm_group, 0, 1)

        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group, 1, 0)

        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group, 1, 1)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group, 2, 0)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)

        ## LDA section

        lda_group = QGroupBox("Linear Discriminant Analysis")
        lda_layout = QVBoxLayout()

        lda_params = self.create_algorithm_group(
            "LDA Parameters",
            {"n_components": "int"}
        )
        lda_layout.addWidget(lda_params)

        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 1, 0)

        # t-SNE section

        tsne_group = QGroupBox("t-SNE")
        tsne_layout = QVBoxLayout()

        tsne_params = self.create_algorithm_group(
            "t-SNE Parameters",
            {"n_components": "int",
             "perplexity": "double"}
        )
        tsne_layout.addWidget(tsne_params)

        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group, 1, 1)

        # UMAP section

        umap_group = QGroupBox("UMAP")
        umap_layout = QVBoxLayout()

        umap_params = self.create_algorithm_group(
            "UMAP Parameters",
            {"n_components": "int",
             "n_neighbors": "int"}
        )
        umap_layout.addWidget(umap_params)

        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 2, 0)

        
        return widget
    
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_gan_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Generator ve Discriminator architecture
        self.gan_epochs_spin = QSpinBox()
        self.gan_epochs_spin.setRange(1, 500)
        self.gan_epochs_spin.setValue(100)

        self.gan_batch_spin = QSpinBox()
        self.gan_batch_spin.setRange(16, 512)
        self.gan_batch_spin.setValue(64)

        train_btn = QPushButton("Train GAN")
        train_btn.clicked.connect(self.train_gan)

        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.gan_epochs_spin)
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.gan_batch_spin)
        layout.addWidget(train_btn)

        return widget

    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        viz_group.setMaximumHeight(400)
        
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        # Real-time training log panel
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Training logs will appear here...")
        viz_layout.addWidget(self.log_text_edit)

        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif param_type == "lineedit":
                widget = QLineEdit()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        mlp_group.setMaximumWidth(640)
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Added layers list
        self.layer_list_widget = QListWidget()
        self.layer_list_widget.setFixedHeight(150)
        mlp_layout.addWidget(QLabel("Added Layers:"))
        mlp_layout.addWidget(self.layer_list_widget)

        # Delete layers with right click
        self.layer_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list_widget.customContextMenuRequested.connect(self.show_layer_context_menu)


        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        # Save / Load buttons
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        mlp_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        mlp_layout.addWidget(load_btn)

        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # Pretrained model selector
        pretrained_label = QLabel("Pretrained Model:")
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(["None", "VGG16", "ResNet50", "MobileNetV2"])
        cnn_layout.addWidget(pretrained_label)
        cnn_layout.addWidget(self.pretrained_combo)

        # Freeze weights
        self.freeze_check = QCheckBox("Freeze Base Model (Feature Extractor)")
        self.freeze_check.setChecked(True)
        cnn_layout.addWidget(self.freeze_check)

        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout", "LSTM", "GRU"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                # L2 regularizer
                l2_label = QLabel("L2 Lambda:")
                l2_input = QDoubleSpinBox()
                l2_input.setRange(0.0, 1.0)
                l2_input.setSingleStep(0.001)
                l2_input.setValue(0.0)
                self.layer_param_inputs["l2"] = l2_input

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
                params_layout.addWidget(l2_label)
                params_layout.addWidget(l2_input)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                # L2 regularizer
                l2_label = QLabel("L2 Lambda:")
                l2_input = QDoubleSpinBox()
                l2_input.setRange(0.0, 1.0)
                l2_input.setSingleStep(0.001)
                l2_input.setValue(0.0)
                self.layer_param_inputs["l2"] = l2_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
                params_layout.addWidget(l2_label)
                params_layout.addWidget(l2_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

            elif layer_type == "MaxPooling2D": 
                pool_size_label = QLabel("Pool Size:")
                pool_size_input = QLineEdit()
                pool_size_input.setText("2, 2")
                self.layer_param_inputs["pool_size"] = pool_size_input
                
                params_layout.addWidget(pool_size_label)
                params_layout.addWidget(pool_size_input)
            
            elif layer_type == "Flatten":
                # Flatten layer has no parameters, just add a label
                notice_label = QLabel("Flatten Layer (no parameters)")
                self.layer_param_inputs["note"] = notice_label
                params_layout.addWidget(notice_label)

            elif layer_type == "LSTM":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1024)
                units_input.setValue(64)
                self.layer_param_inputs["units"] = units_input

                return_seq_checkbox = QCheckBox("Return Sequences")
                self.layer_param_inputs["return_sequences"] = return_seq_checkbox

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["tanh", "relu", "sigmoid"])
                self.layer_param_inputs["activation"] = activation_combo

                # L2 regularizer
                l2_label = QLabel("L2 Lambda:")
                l2_input = QDoubleSpinBox()
                l2_input.setRange(0.0, 1.0)
                l2_input.setSingleStep(0.001)
                l2_input.setValue(0.0)
                self.layer_param_inputs["l2"] = l2_input

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(return_seq_checkbox)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
                params_layout.addWidget(l2_label)
                params_layout.addWidget(l2_input)

            elif layer_type == "GRU":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1024)
                units_input.setValue(64)
                self.layer_param_inputs["units"] = units_input

                return_seq_checkbox = QCheckBox("Return Sequences")
                self.layer_param_inputs["return_sequences"] = return_seq_checkbox

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["tanh", "relu", "sigmoid"])
                self.layer_param_inputs["activation"] = activation_combo

                # L2 regularizer
                l2_label = QLabel("L2 Lambda:")
                l2_input = QDoubleSpinBox()
                l2_input.setRange(0.0, 1.0)
                l2_input.setSingleStep(0.001)
                l2_input.setValue(0.0)
                self.layer_param_inputs["l2"] = l2_input

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(return_seq_checkbox)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
                params_layout.addWidget(l2_label)
                params_layout.addWidget(l2_input)


        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    layer_params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name in ["kernel_size", "pool_size"]:
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            # GUI layer list update
            layer_text = f"{layer_type} - {layer_params}"
            self.layer_list_widget.addItem(layer_text)
            self.status_bar.showMessage(f"Added {layer_type} layer")

            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        # Learning Rate Scheduler
        sched_layout = QHBoxLayout()
        sched_layout.addWidget(QLabel("LR Scheduler:"))
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Step Decay", "Exponential Decay"])
       
        # EarlyStopping
        early_layout = QHBoxLayout()
        self.earlystop_checkbox = QCheckBox("Use EarlyStopping")
        early_layout.addWidget(self.earlystop_checkbox)

        early_layout.addWidget(QLabel("Patience:"))
        self.early_patience_spin = QSpinBox()
        self.early_patience_spin.setRange(1, 100)
        self.early_patience_spin.setValue(5)
        early_layout.addWidget(self.early_patience_spin)

        layout.addLayout(early_layout)

        sched_layout.addWidget(self.scheduler_combo)
        layout.addLayout(sched_layout)

        # Augmentation options
        aug_group = QGroupBox("Image Augmentation")
        aug_layout = QVBoxLayout()

        self.aug_rotation = QCheckBox("Random Rotation")
        self.aug_flip = QCheckBox("Random Flip")
        self.aug_zoom = QCheckBox("Random Zoom")
        self.aug_brightness = QCheckBox("Random Brightness")

        aug_layout.addWidget(self.aug_rotation)
        aug_layout.addWidget(self.aug_flip)
        aug_layout.addWidget(self.aug_zoom)
        aug_layout.addWidget(self.aug_brightness)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

        # Optimizer
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        opt_layout.addWidget(self.optimizer_combo)
        layout.addLayout(opt_layout)



        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()


        # Training parameters (reuse)
        training_params = self.create_training_params_group()
        layout.addWidget(training_params)

        # Train button (shared logic)
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        layout.addWidget(train_btn)

        # Save button
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        layout.addWidget(save_btn)

        # Load button
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        layout.addWidget(load_btn)

        group.setLayout(layout)
        return group



    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()


        # Training parameters (reuse)
        training_params = self.create_training_params_group()
        layout.addWidget(training_params)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        layout.addWidget(train_btn)

        # Save/Load
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        layout.addWidget(save_btn)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        layout.addWidget(load_btn)

        group.setLayout(layout)
        return group

    
    def train_gan(self):

        latent_dim = 100
        epochs = self.gan_epochs_spin.value()
        batch_size = self.gan_batch_spin.value()

        # Basic Generator
        generator = models.Sequential([
            layers.Dense(128, activation="relu", input_dim=latent_dim),
            layers.Dense(784, activation="sigmoid"),
            layers.Reshape((28, 28, 1))
        ])

        # Basic Discriminator
        discriminator = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # GAN (freeze discriminator)
        discriminator.trainable = False
        gan_input = layers.Input(shape=(latent_dim,))
        generated_image = generator(gan_input)
        gan_output = discriminator(generated_image)

        gan = models.Model(gan_input, gan_output)
        gan.compile(optimizer="adam", loss="binary_crossentropy")

        # load MNIST 
        (X_train, _), _ = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype("float32") / 255.0
        X_train = np.expand_dims(X_train, axis=-1)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, real)

            self.status_bar.showMessage(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")
            self.log_message(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
            QApplication.processEvents()
            
            if (epoch + 1) % 10 == 0:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                sample_img = gen_imgs[0].reshape(28, 28)
                ax.imshow(sample_img, cmap='gray')
                ax.set_title(f"Generated Image - Epoch {epoch+1}")
                self.canvas.draw()


            # image generation every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                sample_img = gen_imgs[0].reshape(28, 28)
                ax.imshow(sample_img, cmap='gray')
                ax.set_title(f"Generated Image - Epoch {epoch+1}")
                self.canvas.draw()

                self.log_message(f"Generated image at epoch {epoch+1}")

            self.log_message("GAN training complete.")


    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # create model based on selected pretrained model or custom architecture
            pretrained = self.pretrained_combo.currentText()
            if pretrained != "None":
                model = self.create_pretrained_model(pretrained)
            else:
                model = self.create_neural_network()




            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test

                # reshape for Conv2D if necessary
                if len(X_train.shape) == 3:
                    X_train = X_train[..., np.newaxis]
                    X_test = X_test[..., np.newaxis]

                X_train, y_train = self.create_augmented_input(X_train, tf.keras.utils.to_categorical(self.y_train))
                X_test = tf.cast(X_test, tf.float32) / 255.0
                y_test = tf.keras.utils.to_categorical(self.y_test)


            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer_name = self.optimizer_combo.currentText()
            if optimizer_name == "Adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == "SGD":
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer_name == "RMSprop":
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                self.show_error("Invalid optimizer selected.")
                return

            # Compile the model here (for CNNs or any model that came from pretrained block)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            
            # Callback list
            callbacks = [self.create_progress_callback()]
            callbacks.append(self.create_gradient_logger())


            # EarlyStopping
            if self.earlystop_checkbox.isChecked():
                patience = self.early_patience_spin.value()
                es_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(es_callback)


            # Learning rate scheduler 
            scheduler_type = self.scheduler_combo.currentText()

            if scheduler_type == "Step Decay":
                def step_decay(epoch, lr):
                    drop = 0.5
                    epochs_drop = 5
                    if epoch % epochs_drop == 0 and epoch:
                        return lr * drop
                    return lr
                lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
                callbacks.append(lr_callback)

            elif scheduler_type == "Exponential Decay":
                def exp_decay(epoch, lr):
                    k = 0.1
                    return lr * np.exp(-k * epoch)
                lr_callback = tf.keras.callbacks.LearningRateScheduler(exp_decay)
                callbacks.append(lr_callback)


            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=callbacks)
            
            self.current_model = model

            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]

            self.metrics_text.setText(
                f"Training Loss: {train_loss:.4f}\n"
                f"Validation Loss: {val_loss:.4f}\n"
                f"Training Accuracy: {train_acc:.4f}\n"
                f"Validation Accuracy: {val_acc:.4f}"
)

            
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")



    def create_augmented_input(self, X, y):
        aug_layers = []
            
        if self.aug_rotation.isChecked():
            aug_layers.append(tf.keras.layers.RandomRotation(0.1))
        if self.aug_flip.isChecked():
            aug_layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
        if self.aug_zoom.isChecked():
            aug_layers.append(tf.keras.layers.RandomZoom(0.2))
        if self.aug_brightness.isChecked():
            aug_layers.append(tf.keras.layers.RandomBrightness(0.2))
            
        if aug_layers:
            data_augment = tf.keras.Sequential(aug_layers)
            X = tf.cast(X, tf.float32) / 255.0  # normalize
            X = data_augment(X, training=True)
        else:
            X = tf.cast(X, tf.float32) / 255.0  # just normalize
            
        return X, y

    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                l2_val = params.pop("l2", 0.0)
                if l2_val > 0.0:
                    params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_val)
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                l2_val = params.pop("l2", 0.0)
                if l2_val > 0.0:
                    params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_val)
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    input_shape = self.X_train.shape
                    if len(input_shape) == 3:
                        params['input_shape'] = (*input_shape[1:], 1)
                    else:
                        params['input_shape'] = input_shape[1:]

            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
            elif layer_type == "Conv2D":
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]  # Only for first layer
                model.add(layers.Conv2D(**params))

            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D(**params))

            elif layer_type == "Flatten":
                model.add(layers.Flatten())

            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))

            elif layer_type == "LSTM":
                l2_val = params.pop("l2", 0.0)
                if l2_val > 0.0:
                    params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_val)
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.LSTM(**params))

            elif layer_type == "GRU":
                l2_val = params.pop("l2", 0.0)
                if l2_val > 0.0:
                    params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_val)
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.GRU(**params))


        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model

            
    def create_progress_callback(self):
        return self.ProgressCallbackWithLog(self.progress_bar, self.log_message)

    class ProgressCallbackWithLog(tf.keras.callbacks.Callback):
        def __init__(self, progress_bar, log_fn):
            super().__init__()
            self.progress_bar = progress_bar
            self.log_fn = log_fn

        def on_epoch_end(self, epoch, logs=None):
            progress = int(((epoch + 1) / self.params['epochs']) * 100)
            self.progress_bar.setValue(progress)

            if logs:
                self.log_fn(f"[Epoch {epoch + 1}] "
                            f"Loss: {logs['loss']:.4f}, "
                            f"Val Loss: {logs.get('val_loss', 0):.4f}, "
                            f"Acc: {logs.get('accuracy', 0):.4f}, "
                            f"Val Acc: {logs.get('val_accuracy', 0):.4f}")

    
    def log_message(self, message):
        """Append log messages to real-time log panel."""
        self.log_text_edit.append(message)

        
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        
        self.metrics_text.setText(metrics_text)
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
    
    def read_params(self, model_name, param_widgets):
        """read parameters"""
        
        if model_name == "Support Vector Regression":
            kernel=param_widgets["kernel"].currentText()
            loss_function=param_widgets["loss_function"].currentText()
            C=param_widgets["C"].value()
            epsilon=param_widgets["epsilon"].value()

            from sklearn.svm import SVR
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            return model, loss_function
        
        elif model_name == "Support Vector Machine":
            kernel=param_widgets["kernel"].currentText()
            loss_function=param_widgets["loss_function"].currentText()
            C=param_widgets["C"].value()
            degree=param_widgets["degree"].value()

            from sklearn.svm import SVC

            if loss_function == "Cross-Entropy":
                model = SVC(kernel=kernel, C=C, degree=degree, probability=True)
            else:
                model = SVC(kernel=kernel, C=C, degree=degree, probability=False)

            return model, loss_function
        
        elif model_name == "Logistic Regression":
            C = param_widgets["C"].value()
            max_iter = param_widgets["max_iter"].value()
            multi_class = param_widgets["multi_class"].currentText()
            loss_function = param_widgets["loss_function"].currentText()

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class)
            return model, loss_function
        
        elif model_name == "Linear Regression":
            fit_intercept = param_widgets["fit_intercept"].isChecked()
            normalize = param_widgets["normalize"].isChecked()
            loss_function = param_widgets["loss_function"].currentText()

            from sklearn.linear_model import LinearRegression
            model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
            return model, loss_function


        elif model_name == "Naive Bayes":

            var_smoothing = param_widgets["var_smoothing"].value()
            priors_input = param_widgets["priors"].text()
            
            try:
                if priors_input.strip()=="":
                    priors=None
                else:
                    priors = list(map(float, priors_input.strip("[]").split(",")))
            except:
                raise ValueError("Invalid priors input. Please enter a list of floats.")

            model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            return model, None


        #Dimensionality Reduction Tab

        elif model_name == "PCA Parameters":
            n_components = param_widgets["n_components"].value()
            whiten = param_widgets["whiten"].isChecked()

            from sklearn.decomposition import PCA
            model = PCA(n_components=n_components, whiten=whiten)
            return model, None

        elif model_name == "LDA Parameters":
            n_components = param_widgets["n_components"].value()

            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis(n_components=n_components)
            return model, None
        
        elif model_name == "K-Means Parameters":
            n_clusters = param_widgets["n_clusters"].value()
            max_iter = param_widgets["max_iter"].value()
            n_init = param_widgets["n_init"].value()

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init)
            return model, None
        
        elif model_name == "t-SNE Parameters":
            n_components = param_widgets["n_components"].value()
            perplexity = param_widgets["perplexity"].value()

            from sklearn.manifold import TSNE
            model = TSNE(n_components=n_components, perplexity=perplexity)
            return model, None
        
        elif model_name == "UMAP Parameters":
            n_components = param_widgets["n_components"].value()
            n_neighbors = param_widgets["n_neighbors"].value()

            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
            return model, None
        
            
        
    def train_model(self, model_name, param_widgets):
        
        try:

            if model_name == "PCA Parameters":              
                try:
                    model, _ = self.read_params(model_name, param_widgets)
                    model.fit(self.X_train)

                    variance_ratios = model.explained_variance_ratio_

                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.plot(np.cumsum(variance_ratios), marker='o')
                    ax.set_title("Explained Variance by PCA Components")
                    ax.set_xlabel("Number of Components")
                    ax.set_ylabel("Cumulative Explained Variance")
                    ax.grid(True)
                    self.canvas.draw()

                    self.status_bar.showMessage(f"PCA applied with {model.n_components} components.")

                except Exception as e:
                    self.show_error(f"Error applying PCA: {str(e)}")
                return
            

            if model_name == "LDA Parameters":
                try:
                    model, _ = self.read_params(model_name, param_widgets)
                    X_lda = model.fit_transform(self.X_train, self.y_train)

                   
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    scatter = ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), 
                                        c=self.y_train, cmap='viridis', alpha=0.8)
                    ax.set_title("LDA Projection (1D or 2D)")
                    self.figure.colorbar(scatter)
                    self.canvas.draw()


                    from sklearn.metrics import accuracy_score
                    y_pred = model.predict(self.X_test)
                    accuracy = accuracy_score(self.y_test, y_pred)
                    self.metrics_text.setText(f"Classification Accuracy after LDA: {accuracy:.4f}")

                    self.status_bar.showMessage(f"LDA applied with {model.n_components} components.")

                except Exception as e:
                    self.show_error(f"Error applying LDA: {str(e)}")
                return
            
            
            if model_name == "K-Means Parameters":
                try:
                    model, _ = self.read_params(model_name, param_widgets)
                    model.fit(self.X_train)
                    y_pred = model.predict(self.X_test)

                    # Elbow Method 
                    inertias = []
                    k_range = range(1, 11)
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(self.X_train)
                        inertias.append(km.inertia_)

                    self.figure.clear()
                    ax1 = self.figure.add_subplot(211)
                    ax1.plot(k_range, inertias, marker='o')
                    ax1.set_title("Elbow Method: Inertia vs. k")
                    ax1.set_xlabel("Number of Clusters (k)")
                    ax1.set_ylabel("Inertia")
                    ax1.grid(True)

                    # Silhouette Score
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(self.X_test, y_pred)

                    ax2 = self.figure.add_subplot(212)
                    scatter = ax2.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap='viridis')
                    ax2.set_title("k-Means Clustering Result")
                    self.figure.colorbar(scatter)

                    self.figure.tight_layout()
                    self.canvas.draw()

                    self.metrics_text.setText(f"Silhouette Score: {silhouette:.4f}")
                    self.status_bar.showMessage(f"k-Means trained with k={model.n_clusters}, silhouette={silhouette:.4f}")

                except Exception as e:
                    self.show_error(f"Error training k-Means: {str(e)}")
                return
            
            if model_name == "t-SNE Parameters":
                try:
                    model, _ = self.read_params(model_name, param_widgets)
                    X_tsne = model.fit_transform(self.X_test)

                    if self.use_plotly_checkbox.isChecked():
                        df = pd.DataFrame(X_tsne, columns=["Dim 1", "Dim 2"] if X_tsne.shape[1] == 2 else ["Dim 1", "Dim 2", "Dim 3"])
                        df["Label"] = self.y_test.astype(str)

                        if X_tsne.shape[1] == 3:
                            fig = px.scatter_3d(df, x="Dim 1", y="Dim 2", z="Dim 3", color="Label",
                                                title="t-SNE Projection (3D)")
                        else:
                            fig = px.scatter(df, x="Dim 1", y="Dim 2", color="Label",
                                            title="t-SNE Projection (2D)")
                        fig.show()

                    else:
                        self.figure.clear()
                        if X_tsne.shape[1] == 2:
                            ax = self.figure.add_subplot(111)
                            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                                                c=self.y_test, cmap='viridis', alpha=0.8)
                            ax.set_title("t-SNE Projection (2D)")
                            self.figure.colorbar(scatter)

                        elif X_tsne.shape[1] == 3:
                            from mpl_toolkits.mplot3d import Axes3D
                            ax = self.figure.add_subplot(111, projection='3d')
                            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                                                c=self.y_test, cmap='viridis', alpha=0.8)
                            ax.set_title("t-SNE Projection (3D)")

                        self.canvas.draw()

                    self.status_bar.showMessage("t-SNE projection complete.")
                    self.metrics_text.setText("t-SNE projection completed successfully.")

                except Exception as e:
                    self.show_error(f"Error applying t-SNE: {str(e)}")
                return

            
            if model_name == "UMAP Parameters":
                try:
                    model, _ = self.read_params(model_name, param_widgets)
                    X_umap = model.fit_transform(self.X_test)

                    if self.use_plotly_checkbox.isChecked():
                        df = pd.DataFrame(X_umap, columns=["Dim 1", "Dim 2"] if X_umap.shape[1] == 2 else ["Dim 1", "Dim 2", "Dim 3"])
                        df["Label"] = self.y_test.astype(str)

                        if X_umap.shape[1] == 3:
                            fig = px.scatter_3d(df, x="Dim 1", y="Dim 2", z="Dim 3", color="Label",
                                                title="UMAP Projection (3D)")
                        else:
                            fig = px.scatter(df, x="Dim 1", y="Dim 2", color="Label",
                                            title="UMAP Projection (2D)")
                        fig.show()

                    else:
                        self.figure.clear()
                        if X_umap.shape[1] == 2:
                            ax = self.figure.add_subplot(111)
                            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                                                c=self.y_test, cmap='viridis', alpha=0.8)
                            ax.set_title("UMAP Projection (2D)")
                            self.figure.colorbar(scatter)

                        elif X_umap.shape[1] == 3:
                            from mpl_toolkits.mplot3d import Axes3D
                            ax = self.figure.add_subplot(111, projection='3d')
                            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                                                c=self.y_test, cmap='viridis', alpha=0.8)
                            ax.set_title("UMAP Projection (3D)")

                        self.canvas.draw()

                    self.status_bar.showMessage("UMAP projection complete.")
                    self.metrics_text.setText("UMAP is unsupervised. No accuracy metric.")

                except Exception as e:
                    self.show_error(f"Error applying UMAP: {str(e)}")
                return



            
            # Check if model is compatible with target type(at first, did not realise base gui already does this)
            
            from sklearn.utils.multiclass import type_of_target
            target_type = type_of_target(self.y_train)

            if model_name == "Support Vector Regression" and target_type not in ["continuous", "continuous-multioutput"]:
                raise ValueError("SVR can only be used with continuous numeric targets.")

            if model_name == "SVM" and target_type in ["continuous", "continuous-multioutput"]:
                raise ValueError("SVM cannot be used with continuous targets.")

            if model_name == "Linear Regression" and target_type not in ["continuous", "continuous-multioutput"]:
                raise ValueError("Linear Regression requires continuous numeric targets.")

            if model_name == "Logistic Regression" and target_type not in ["binary", "multiclass"]:
                raise ValueError("Logistic Regression requires categorical targets (binary or multiclass).")

            if model_name == "Naive Bayes" and target_type not in ["binary", "multiclass"]:
                raise ValueError("Naive Bayes requires categorical targets (binary or multiclass).")
            
            
            # train the model

            model, loss_function = self.read_params(model_name, param_widgets)

            if self.kfold_checkbox.isChecked():
                from sklearn.model_selection import cross_val_score
                k = self.kfold_spin.value()

                if loss_function in ["MSE", "MAE"]:
                    from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
                    if loss_function == "MSE":
                        scorer = make_scorer(mean_squared_error)
                    else:
                        scorer = make_scorer(mean_absolute_error)

                else:
                    scorer = "accuracy"

                scores = cross_val_score(model, self.X_train, self.y_train, cv=k, scoring=scorer)

                mean_score = np.mean(scores)
                std_score = np.std(scores)

                self.metrics_text.setText(
                    f"k-Fold Cross-Validation ({k}-fold)\n"
                    f"Mean Score: {mean_score:.4f}\n"
                    f"Std Deviation: {std_score:.4f}"
                )
                self.status_bar.showMessage(f"k-Fold CV complete. Mean={mean_score:.4f}, Std={std_score:.4f}")
                return



            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from sklearn.linear_model import HuberRegressor

            if loss_function == "MSE":
                loss = mean_squared_error(self.y_test, y_pred)
            elif loss_function == "MAE":
                loss = mean_absolute_error(self.y_test, y_pred)
            elif loss_function == "Huber":
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor()
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                loss = mean_absolute_error(self.y_test, y_pred)
            elif loss_function == "Cross-Entropy":
                from sklearn.metrics import log_loss
                try:
                    y_prob = model.predict_proba(self.X_test)
                    loss = log_loss(self.y_test, y_prob)
                except:
                    raise ValueError("Cross-Entropy loss requires a model with predict_proba method.")
            elif loss_function == "Hinge":
                from sklearn.metrics import hinge_loss
                try:
                    if hasattr(model, "decision_function"):
                        y_score = model.decision_function(self.X_test)
                        y_true = np.where(self.y_test == 1, 1, -1)
                        loss = hinge_loss(y_true, y_score)
                    else:
                        raise ValueError("Model does not support decision_function. Set probability=False.")
                except:
                    raise ValueError("Hinge loss requires a model with decision_function method.")


            else:
                loss = None


            self.current_model = model
            self.update_metrics(y_pred)
            self.update_visualization(y_pred)

            if loss is not None:
                self.status_bar.showMessage(f"Trained {model_name} model with loss: {loss:.4f}")
            else:
                self.status_bar.showMessage(f"Trained {model_name} model.")

        except Exception as e:
            self.show_error(f"Error training {model_name} model: {str(e)}")


    def save_model(self):
        try:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "HDF5 Files (*.h5)")
            if file_name:
                self.current_model.save(file_name)
                self.status_bar.showMessage(f"Model saved to {file_name}")
        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")

    def load_model(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "HDF5 Files (*.h5)")
            if file_name:
                self.current_model = tf.keras.models.load_model(file_name)
                self.status_bar.showMessage(f"Model loaded from {file_name}")
        except Exception as e:
            self.show_error(f"Error loading model: {str(e)}")

    def create_pretrained_model(self, model_name):
        input_shape = self.X_train.shape[1:]
        num_classes = len(np.unique(self.y_train))

        if model_name == "VGG16":
            base_model = tf.keras.applications.VGG16(
                input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == "MobileNetV2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            raise ValueError("Unknown pretrained model selected.")

        base_model.trainable = not self.freeze_check.isChecked()

        model = tf.keras.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model
    
    def create_gradient_logger(self):
        class GradientLogger(tf.keras.callbacks.Callback):
            def __init__(self, canvas, figure, X_val, y_val):
                super().__init__()
                self.canvas = canvas
                self.figure = figure
                self.X_val = X_val
                self.y_val = y_val

            def on_epoch_end(self, epoch, logs=None):
                grads = []

                x_sample = self.X_val[:1]
                y_sample = self.y_val[:1]

                with tf.GradientTape() as tape:
                    y_pred = self.model(x_sample, training=True)
                    loss = self.model.compiled_loss(y_sample, y_pred)

                gradients = tape.gradient(loss, self.model.trainable_variables)

                for g in gradients:
                    if g is not None:
                        grads.append(tf.reduce_mean(tf.abs(g)).numpy())

                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.hist(grads, bins=15, color="dodgerblue", edgecolor="black", alpha=0.8)
                ax.set_title(f"Gradient Histogram - Epoch {epoch+1}")
                ax.set_xlabel("Mean Absolute Gradient")
                ax.set_ylabel("Layer Count")
                self.canvas.draw()

        
        X_val = tf.cast(self.X_test, tf.float32) / 255.0
        y_val = tf.keras.utils.to_categorical(self.y_test)

        return GradientLogger(self.canvas, self.figure, X_val, y_val)

    
    def show_layer_context_menu(self, pos):
        menu = QMenu(self.layer_list_widget)

        remove_action = menu.addAction("Remove Selected Layer")
        action = menu.exec(self.layer_list_widget.mapToGlobal(pos))

        if action == remove_action:
            selected_items = self.layer_list_widget.selectedItems()
            for item in selected_items:
                row = self.layer_list_widget.row(item)
                self.layer_list_widget.takeItem(row)  # image
                del self.layer_config[row]            # layer_config 
                self.status_bar.showMessage(f"Removed layer at index {row}")
 



def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
