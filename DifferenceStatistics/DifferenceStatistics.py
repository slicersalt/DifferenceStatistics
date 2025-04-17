from __future__ import annotations

import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Optional


import vtk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget, ScriptedLoadableModuleTest,
)


#
# DifferenceStatistics
#

class DifferenceStatistics(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Statistics on Object Differences"
    self.parent.categories = ["Shape Analysis"]
    self.parent.dependencies = []
    self.parent.contributors = ["Kedar Madi (Virginia Tech), Jared Vicory (Kitware)"]
    self.parent.helpText = (
      "Compute statistics on the difference between objects at different timepoints"
    )
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = (
      "This file was originally developed by Jean-Christophe Fillion-Robin, "
      "Kitware Inc., Andras Lasso, PerkLab, and Steve Pieper, Isomics, Inc. "
      "and was partially funded by NIH grant 3P41RR013218-12S1."
    )


#
# DifferenceStatisticsWidget
#

class DifferenceStatisticsWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic: Optional[DifferenceStatisticsLogic] = None

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer)
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/DifferenceStatistics.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.logic = DifferenceStatisticsLogic()

    self.ui.ApplyButton.connect('clicked(bool)', self.onApplyButton)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      self.logic.run(
        Path(self.ui.InputCSV.currentPath),
        Path(self.ui.TemplateMesh.currentPath),
        Path(self.ui.OutputDirectory.directory)
      )
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: {}".format(e))
      import traceback
      traceback.print_exc()


#
# DifferenceStatisticsLogic
#

class DifferenceStatisticsLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def run(self, inputCSV: Path, template:Path, output: Path):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputCSV:
    :param template:
    :param output:
    """
    if not inputCSV.exists():
      raise ValueError('Input CSV is not valid.')

    if output.exists() and not output.is_dir():
      raise ValueError('Output directory is not valid.')

    logging.info('Processing started')

    # Compute differences between timepoints and create difference files

    # Load the CSV file with headers
    df = pd.read_csv(inputCSV)

    # Set column 1 (Timepoint 1) as 'tp1_subjects' and column 2 (Timepoint 2) as 'tp2_subjects'
    tp1_subjects = df['Timepoint 1']
    tp2_subjects = df['Timepoint 2']

    # Remaining columns should be covariates
    covs = df.drop(['Timepoint 1','Timepoint 2'], axis=1)
    covariate_names = covs.columns.tolist()
    names = []

    logging.info('Computing pairwise differences')

    for index, (tp1, tp2) in enumerate(zip(tp1_subjects, tp2_subjects)):
      try:
        # Load models
        _, tp1_name = os.path.split(tp1)
        tp1_name, _ = os.path.splitext(tp1_name)
        model1 = slicer.util.loadModel(tp1)

        _, tp2_name = os.path.split(tp2)
        tp2_name, _ = os.path.splitext(tp2_name)
        model2 = slicer.util.loadModel(tp2)

        out_name = tp2_name + '-' + tp1_name

        if not model1 or not model2:
          raise Exception(f"Skipped {tp1} and {tp2} due to failure in loading models.")

        # Run ModelToModel distance
        output_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",out_name)

        params = {
            'vtkFile1': model1.GetID(),
            'vtkFile2': model2.GetID(), 
            'vtkOutput': output_node.GetID(), 
            'distanceType': 'corresponding_point_to_point', 
            'targetInFields': False
            }
        slicer.cli.runSync(slicer.modules.modeltomodeldistance, None, parameters=params)

        # Replace points with difference vectors
        magNormVectors = slicer.util.arrayFromModelPointData(output_node, 'MagNormVector')
        output_node.GetPolyData().GetPoints().SetData(vtk.util.numpy_support.numpy_to_vtk(magNormVectors))

        output_model_fullpath = (output / f"{out_name}.vtk")
        slicer.util.saveNode(output_node, str(output_model_fullpath))
        names.append(str(output_model_fullpath))
      except Exception as e:
        print(f"An error occurred with subject {index}: {e}")
        return
    
    covs.insert(0,'VTK File',names)
    out_csv_fullpath = (output / "mfsdaInputFiles.csv")
    covs.to_csv(str(out_csv_fullpath),index=False)

    # Run difference files through MFSDA

    logging.info('Computing statistics')

    slicer.util.selectModule(slicer.modules.mfsda)
    slicer.util.selectModule(slicer.modules.differencestatistics)
    
    slicer.modules.MFSDAWidget.lineEdit_pshape.setCurrentPath(str(template))
    slicer.modules.MFSDAWidget.lineEdit_output.directory = str(output)
    slicer.modules.MFSDAWidget.lineEdit_csv.setCurrentPath(str(out_csv_fullpath))

    logging.info('Processing completed')


#
# DifferenceStatisticsTest
#

class DifferenceStatisticsTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_DifferenceStatistics1()

  def test_DifferenceStatistics1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    import tempfile
    import os

    logic = DifferenceStatisticsLogic()

    with tempfile.TemporaryDirectory() as tempdir:
      tempdir = Path(tempdir)

      content1 = os.urandom(32)
      content2 = os.urandom(32)

      data = tempdir / 'data'
      data.mkdir()
      (data / 'file').write_bytes(content1)
      (data / 'sub').mkdir()
      (data / 'sub' / 'file').write_bytes(content2)

      output = tempdir / 'output'

      logic.run(data, output)

      self.assertTrue(output.exists())
      self.assertTrue((output / 'file').exists())
      self.assertEqual((output / 'file').read_bytes(), content1)

      self.assertTrue((output / 'sub').exists())
      self.assertTrue((output / 'file').exists())
      self.assertEqual((output / 'sub' / 'file').read_bytes(), content2)

    self.delayDisplay('Test passed')
