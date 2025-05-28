import os
import logging
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest
)
from qt import QUiLoader, QFile, QIODevice, QPushButton, QTextEdit, QCheckBox
from slicer import qMRMLNodeComboBox, vtkMRMLLabelMapVolumeNode
import numpy as np
from vtk.util import numpy_support
import vtk
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import nibabel as nib


def getResourcePath(path):
    return os.path.join(os.path.dirname(__file__), 'Resources', path)


class StrokeDetectionModule(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Stroke Detection"
        self.parent.categories = ["Classification and Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Furkan Akyel (Inonu University)"]
        self.parent.helpText = (
            "Use the classifier and segmenter models for stroke analysis."
        )
        self.parent.acknowledgementText = (
            "Classifier and Segmenter models are used for slice-level classification and segmentation."
        )


class StrokeDetectionModuleWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        # Load UI
        uiPath = getResourcePath(os.path.join('UI', 'StrokeDetectionModule.ui'))
        uiFile = QFile(uiPath)
        if not uiFile.open(QIODevice.ReadOnly):
            logging.error(f"Cannot open UI file: {uiPath}")
            return
        loader = QUiLoader()
        self.ui = loader.load(uiFile)
        uiFile.close()
        if not self.ui:
            logging.error(f"Failed to load UI: {uiPath}")
            return
        self.layout.addWidget(self.ui)

        # Node selectors
        self.inputSelector = self.ui.findChild(qMRMLNodeComboBox, 'inputSelector')
        self.maskSelector = self.ui.findChild(qMRMLNodeComboBox, 'maskSelector')
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.maskSelector.setMRMLScene(slicer.mrmlScene)

        # Butonlar
        self.classifyButton = self.ui.findChild(QPushButton, 'classifyButton')
        self.segmentButton = self.ui.findChild(QPushButton, 'pushButton')
        self.classifyButton.clicked.connect(self.onClassify)
        self.segmentButton.clicked.connect(self.onSegment)
        self.segmentButton.text = "Bölütle"
        self.aisdSegButton = QPushButton("AISD Verisetini Segmente Et")
        self.ui.layout().insertWidget(5, self.aisdSegButton)
        self.aisdSegButton.clicked.connect(self.onAISDSegmentation)

        # Yeni AISD Butonu
        self.aisdButton = QPushButton("AISD Verisetini Sınıflandır")
        self.ui.layout().insertWidget(4, self.aisdButton)
        self.aisdButton.clicked.connect(self.onAISDClassification)

        # Diğer UI elementleri
        self.show3DCheckbox = QCheckBox("Show 3D")
        self.ui.layout().insertWidget(3, self.show3DCheckbox)
        self.show3DCheckbox.stateChanged.connect(self.update3DDisplay)
        self.resultText = self.ui.findChild(QTextEdit, 'confusionMatrixTextEdit')
        self.currentMaskNode = None
        self.currentSegNode = None

    def onAISDClassification(self):
        self.aisdButton.enabled = False
        self.resultText.clear()
        self.resultText.append("Processing AISD dataset...")
        
        try:
            logic = StrokeDetectionModuleLogic()
            cm, accuracy, report = logic.processAISDDataset()
            
            self.resultText.clear()
            self.resultText.append("=== AISD Dataset Results ===")
            self.resultText.append("\nConfusion Matrix:")
            self.resultText.append(str(cm))
            self.resultText.append(f"\nAccuracy: {accuracy:.4f}")
            self.resultText.append("\nClassification Report:")
            self.resultText.append(report)
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")
        finally:
            self.aisdButton.enabled = True

    def onAISDSegmentation(self):
        self.aisdSegButton.enabled = False
        self.resultText.clear()
        self.resultText.append("AISD veri seti segmentasyonu işleniyor...")
        
        try:
            logic = StrokeDetectionModuleLogic()
            patient_scores, avg_dice = logic.processAISDSegmentation()
            
            self.resultText.clear()
            self.resultText.append("=== AISD Segmentasyon Sonuçları ===")
            for patient, dice in patient_scores.items():
                self.resultText.append(f"{patient}: Dice={dice:.4f}")
            self.resultText.append(f"\nOrtalama Dice Skoru: {avg_dice:.4f}")
            
        except Exception as e:
            slicer.util.errorDisplay(f"Hata: {str(e)}")
        finally:
            self.aisdSegButton.enabled = True

    def onClassify(self):
        inputNode = self.inputSelector.currentNode()
        maskNode = self.maskSelector.currentNode() if self.maskSelector else None
        if not inputNode:
            slicer.util.errorDisplay("Please select an input volume.")
            return
        self.classifyButton.enabled = False
        logic = StrokeDetectionModuleLogic()
        lines = logic.runClassification(inputNode, maskNode)
        self.resultText.clear()
        for line in lines:
            self.resultText.append(line)
        slicer.util.delayDisplay("Classification complete.")
        self.classifyButton.enabled = True

    def onSegment(self):
        inputNode = self.inputSelector.currentNode()
        maskNode = self.maskSelector.currentNode() if self.maskSelector else None
        
        if not inputNode:
            slicer.util.errorDisplay("Please select an input volume.")
            return
        
        self.segmentButton.enabled = False
        logic = StrokeDetectionModuleLogic()
        
        # Segmentasyonu çalıştır
        segNode = logic.runSegmentation(inputNode)
        if segNode is None:
            slicer.util.errorDisplay("Segmentation failed!")
            self.segmentButton.enabled = True
            return
        
        self.currentSegNode = segNode
        
        # Metrikleri hesapla ve göster
        if maskNode:
            self.currentMaskNode = maskNode
            dice, cm, slice_metrics = logic.computeMetrics(segNode, maskNode)
            
            self.resultText.clear()
            self.resultText.append(f"Overall Dice Score: {dice:.4f}")
            self.resultText.append("Confusion Matrix:")
            self.resultText.append(f"\tTN: {cm[0][0]}, FP: {cm[0][1]}")
            self.resultText.append(f"\tFN: {cm[1][0]}, TP: {cm[1][1]}")
            
            self.resultText.append("\nSlice-wise Metrics:")
            for z, slice_dice in slice_metrics:
                self.resultText.append(f"Slice {z}: Dice={slice_dice:.4f}")
        
        # 3D görünümü güncelle
        self.update3DDisplay()
        slicer.util.delayDisplay("Segmentation complete.")
        self.segmentButton.enabled = True

    def update3DDisplay(self):
        show_3d = self.show3DCheckbox.isChecked()
        
        # Segmentasyon görünümü
        if self.currentSegNode:
            segDispNode = self.currentSegNode.GetDisplayNode()
            if segDispNode:
                segDispNode.SetVisibility(show_3d)
                segDispNode.SetOpacity(0.7)  # Yarı saydam
                segDispNode.SetColor(1, 0, 0)  # Kırmızı
        
        # Maske görünümü
        if self.currentMaskNode:
            maskDispNode = self.currentMaskNode.GetDisplayNode()
            if maskDispNode:
                maskDispNode.SetVisibility(show_3d)
                maskDispNode.SetOpacity(0.5)  # Daha saydam
                maskDispNode.SetColor(0, 1, 0)  # Yeşil
        
        # 3D görünümü yenile
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()



class StrokeDetectionModuleLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clsModel = None
        self.segModel = None
        self.loadModels()

    def loadModels(self):
        """Model yükleme ve hazırlama"""
        clsPath = getResourcePath(os.path.join('Models', 'classifier.pt'))
        segPath = getResourcePath(os.path.join('Models', 'segmenter.pt'))
        
        try:
            # Model yüklemelerini try-except içine al
            self.clsModel = torch.jit.load(clsPath, map_location=self.device).eval()
            self.segModel = torch.jit.load(segPath, map_location=self.device).eval()
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            slicer.util.errorDisplay(f"Model yükleme hatası: {str(e)}")

    def preprocessSlice(self, sliceArr):
        """Slice önişleme ve tensor dönüşümü"""
        if np.all(sliceArr == 0):
            return None  # Boş slice'ları atla
            
        tensor = torch.from_numpy(sliceArr.astype(np.float32)).to(self.device)
        
        # Normalizasyon iyileştirmeleri
        t_min = tensor.min()
        t_max = tensor.max()
        if (t_max - t_min) == 0:
            tensor = torch.zeros_like(tensor)
        else:
            tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
            
        return tensor.unsqueeze(0).unsqueeze(0)

    def processAISDDataset(self):
        """AISD veri setini toplu işleme"""
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        
        dataset_path = getResourcePath('AISD')
        image_dir = os.path.join(dataset_path, 'image')
        mask_dir = os.path.join(dataset_path, 'mask')

        all_preds = []
        all_true = []
        processed_files = 0

        try:
            for filename in os.listdir(image_dir):
                if not filename.endswith('.nii.gz'):
                    continue

                img_path = os.path.join(image_dir, filename)
                mask_path = os.path.join(mask_dir, filename)
                
                if not os.path.exists(mask_path):
                    logging.warning(f"{filename} için maske bulunamadı")
                    continue

                try:
                    # NIfTI veri yükleme
                    img = nib.load(img_path).get_fdata()
                    mask = nib.load(mask_path).get_fdata()
                    
                    # Boyut kontrolü
                    if img.shape != mask.shape:
                        logging.error(f"{filename} boyut uyumsuzluğu")
                        continue
                        
                    # Slice işleme
                    for z in range(img.shape[2]):
                        slice_img = img[:, :, z]
                        slice_mask = mask[:, :, z]
                        
                        # Önişleme
                        tensor = self.preprocessSlice(slice_img)
                        if tensor is None:
                            continue  # Boş slice'ı atla
                            
                        # Tahmin
                        with torch.no_grad():
                            output = self.clsModel(tensor)
                            
                            # Model çıktı tipine göre işlem
                            if output.shape[1] == 1:  # Binary classification
                                pred = (torch.sigmoid(output) > 0.5).int().item()
                            else:  # Multi-class
                                pred = torch.argmax(F.softmax(output, dim=1)).item()
                                
                        # Etiket
                        true_label = 1 if np.any(slice_mask > 0) else 0
                        
                        all_preds.append(pred)
                        all_true.append(true_label)
                        
                    processed_files += 1
                    
                except Exception as e:
                    logging.error(f"{filename} işlenirken hata: {str(e)}")
                    continue

            # Sonuç kontrolleri
            if not all_true:
                raise ValueError("Hiç veri işlenemedi")
                
            if len(all_true) != len(all_preds):
                raise ValueError("Etiket ve tahmin sayısı uyuşmuyor")

            # Metrik hesaplama
            cm = confusion_matrix(all_true, all_preds)
            accuracy = accuracy_score(all_true, all_preds)
            report = classification_report(all_true, all_preds)
            
            return cm, accuracy, report

        except Exception as e:
            logging.error(f"Dataset processing failed: {str(e)}")
            raise

    def volumeToArray(self, node):
        """Convert MRML volume node to numpy array"""
        img = node.GetImageData()
        scalars = img.GetPointData().GetScalars()
        arr = numpy_support.vtk_to_numpy(scalars)
        return arr.reshape(img.GetDimensions(), order='F')

    def preprocessSlice(self, sliceArr):
        # Rest of the code remains the same...
        tensor = torch.from_numpy(sliceArr).float().to(self.device)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        return tensor.unsqueeze(0).unsqueeze(0)

    def runClassification(self, inputVol, maskVol=None):
        if self.clsModel is None:
            return ["Classifier model not loaded."]
        inp = self.volumeToArray(inputVol)
        zDim = inp.shape[2]
        results = []
        for z in range(zDim):
            tensor = self.preprocessSlice(inp[:, :, z])
            with torch.no_grad():
                out = self.clsModel(tensor).cpu()
                if out.shape[1] == 1:
                    prob = torch.sigmoid(out).item()
                    pred = int(prob > 0.5)
                else:
                    pred = int(torch.argmax(F.softmax(out, dim=1)).item())
            if maskVol:
                maskArr = self.volumeToArray(maskVol)
                trueLabel = 1 if np.any(maskArr[:, :, z] > 0) else 0
                results.append(f"Slice {z}: Pred={pred}, True={trueLabel}")
            else:
                results.append(f"Slice {z}: Pred={pred}")
        return results

    def runSegmentation(self, inputVol):
        if self.segModel is None:
            return None
        
        arr = self.volumeToArray(inputVol)
        dims = arr.shape
        segArr = np.zeros(dims, dtype=np.uint8)
        
        try:
            for z in range(dims[2]):
                tensor = self.preprocessSlice(arr[:, :, z])
                with torch.no_grad():
                    out = self.segModel(tensor)
                    mask = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                segArr[:, :, z] = mask * 255
        except Exception as e:
            logging.error(f"Segmentation inference error: {e}")
            return None
        
        # Tek bir segmentasyon dosyası oluştur
        imgData = vtk.vtkImageData()
        imgData.SetDimensions(dims)
        vtkScalars = numpy_support.numpy_to_vtk(segArr.ravel(order='F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        imgData.GetPointData().SetScalars(vtkScalars)
        
        segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'Stroke_Segmentation')
        segNode.SetAndObserveImageData(imgData)
        
        # Görüntüleme ayarları
        dispNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeDisplayNode')
        dispNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeLabels')
        dispNode.SetDefaultColorMap()
        segNode.AddAndObserveDisplayNodeID(dispNode.GetID())
        
        return segNode

    def computeMetrics(self, segNode, maskNode):
        segArr = self.volumeToArray(segNode)
        maskArr = self.volumeToArray(maskNode)
        
        # Tüm hacim için metrikler
        seg_flat = (segArr > 0).astype(np.uint8).ravel()
        mask_flat = (maskArr > 0).astype(np.uint8).ravel()
        
        # Confusion matrix
        tp = int(((seg_flat == 1) & (mask_flat == 1)).sum())
        fp = int(((seg_flat == 1) & (mask_flat == 0)).sum())
        fn = int(((seg_flat == 0) & (mask_flat == 1)).sum())
        tn = int(((seg_flat == 0) & (mask_flat == 0)).sum())
        
        # Dice skoru
        dice = 2.0 * tp / (2 * tp + fp + fn + 1e-8)
        
        # Dilim bazlı metrikler
        slice_metrics = []
        for z in range(segArr.shape[2]):
            seg_slice = (segArr[:, :, z] > 0).astype(np.uint8)
            mask_slice = (maskArr[:, :, z] > 0).astype(np.uint8)
            
            slice_tp = int(((seg_slice == 1) & (mask_slice == 1)).sum())
            slice_fp = int(((seg_slice == 1) & (mask_slice == 0)).sum())
            slice_fn = int(((seg_slice == 0) & (mask_slice == 1)).sum())
            
            slice_dice = 2.0 * slice_tp / (2 * slice_tp + slice_fp + slice_fn + 1e-8) if (2*slice_tp + slice_fp + slice_fn) > 0 else 0.0
            slice_metrics.append((z, slice_dice))
        
        return dice, [[tn, fp], [fn, tp]], slice_metrics
    
    def processAISDSegmentation(self):
        """Process entire AISD dataset for segmentation and compute Dice scores"""
        dataset_path = getResourcePath('AISD')
        image_dir = os.path.join(dataset_path, 'image')
        mask_dir = os.path.join(dataset_path, 'mask')

        patient_scores = {}
        total_dice = 0.0
        num_patients = 0

        for filename in os.listdir(image_dir):
            if not filename.endswith('.nii.gz'):
                continue

            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            if not os.path.exists(mask_path):
                logging.warning(f"{filename} için maske bulunamadı")
                continue

            try:
                # NIfTI dosyalarını yükle
                img = nib.load(img_path).get_fdata()
                mask = nib.load(mask_path).get_fdata()

                if img.shape != mask.shape:
                    logging.error(f"{filename} boyut uyumsuzluğu")
                    continue

                # Segmentasyon için tahminleri sakla
                seg_pred = np.zeros_like(img, dtype=np.uint8)

                # Her dilimi işle
                for z in range(img.shape[2]):
                    slice_img = img[:, :, z]
                    
                    # Önişleme
                    tensor = self.preprocessSlice(slice_img)
                    if tensor is None:
                        continue  # Boş slice'ı atla
                    
                    # Tahmin
                    with torch.no_grad():
                        output = self.segModel(tensor)
                        pred_slice = (torch.sigmoid(output).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    
                    seg_pred[:, :, z] = pred_slice * 1  # 1 ve 0'lardan oluşan maske

                # Dice skorunu hesapla
                seg_flat = (seg_pred > 0.5).astype(np.uint8).ravel()
                mask_flat = (mask > 0.5).astype(np.uint8).ravel()

                tp = np.sum((seg_flat == 1) & (mask_flat == 1))
                fp = np.sum((seg_flat == 1) & (mask_flat == 0))
                fn = np.sum((seg_flat == 0) & (mask_flat == 1))

                dice = 2.0 * tp / (2 * tp + fp + fn + 1e-8)
                patient_scores[filename] = dice
                total_dice += dice
                num_patients += 1

            except Exception as e:
                logging.error(f"{filename} işlenirken hata: {str(e)}")
                continue

        if num_patients == 0:
            raise ValueError("İşlenen hasta bulunamadı")

        avg_dice = total_dice / num_patients
        return patient_scores, avg_dice

class StrokeDetectionModuleTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.testLogic()

    def testLogic(self):
        import vtk
        dims = (4,4,2)
        arr = np.zeros(dims, dtype=np.uint8)
        mask = np.zeros(dims, dtype=np.uint8)
        mask[:, :, 1] = 255
        def createVol(a, name, label=False):
            img = vtk.vtkImageData()
            img.SetDimensions(a.shape)
            vtkScalars = numpy_support.numpy_to_vtk(a.ravel(order='F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            img.GetPointData().SetScalars(vtkScalars)
            cls = 'vtkMRMLLabelMapVolumeNode' if label else 'vtkMRMLScalarVolumeNode'
            node = slicer.mrmlScene.AddNewNodeByClass(cls, name)
            node.SetAndObserveImageData(img)
            return node
        inNode = createVol(arr, 'In')
        maskNode = createVol(mask, 'Mask', label=True)
        logic = StrokeDetectionModuleLogic()
        if logic.clsModel is None or logic.segModel is None:
            self.skipTest("Models not loaded, skipping tests.")
        resCls = logic.runClassification(inNode, maskNode)
        segNode = logic.runSegmentation(inNode)
        assert segNode is not None
        dice, cm = logic.computeMetrics(segNode, maskNode)
        assert len(resCls) == dims[2]
        assert 0.0 <= dice <= 1.0
        logging.info('All tests passed')