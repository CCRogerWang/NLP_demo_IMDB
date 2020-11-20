//
//  ViewController.swift
//  MLNLP
//
//  Created by Roger on 2020/11/15.
//

import Cocoa
import CreateML
import Foundation

import NaturalLanguage
import CoreML

class ViewController: NSViewController {

    
    let saveModelPath = "/Users/roger.wang/Documents/Workspace/nlp_demo_IMDB/IMDBReviewClassifier.mlmodel"

    var trainingModel: MLTextClassifier?
    
    
    @IBOutlet var inputTextView: NSTextView!
    @IBOutlet weak var resultLabel: NSTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        inputTextView.font = NSFont.systemFont(ofSize: 24)
        
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }

    // MARK: - Action
    @IBAction func tapRun(_ sender: NSButton) {
        print("tapRun")
        runML()
    }

    @IBAction func tapSave(_ sender: NSButton) {
        print("\n===== START SAVE =====\n")
        saveModel()
        print("\n===== END SAVE =====\n")
        
    }
    
    @IBAction func predict(_ sender: Any) {
        let mlModel = try! IMDBReviewClassifier(configuration: MLModelConfiguration()).model
                
        let sentimentPredictor = try! NLModel(mlModel: mlModel)
        let text = inputTextView.string
        resultLabel.stringValue = "\(text) is \(sentimentPredictor.predictedLabel(for: text) ?? "unknow")"

    }
    
    // MARK: - private
    private func runML() {
        guard let datasetFilePath = Bundle.main.url(forResource: "reviews", withExtension: "json") else {

            fatalError("train file path is nil")
        }

        let dataset = try! MLDataTable(contentsOf: datasetFilePath)

        print("random split 0.8")
        
        let (trainingData, testingData) = dataset.randomSplit(by: 0.8, seed: 5)

        
        print("\n===== START TRAIN =====\n")
        
        
        trainingModel = try! MLTextClassifier(trainingData: trainingData, textColumn: "SentimentText", labelColumn: "Sentiment")
        
        // Training accuracy as a percentage
        let trainingAccuracy = (1.0 - trainingModel!.trainingMetrics.classificationError) * 100
        print("trainingAccuracy = \(trainingAccuracy)")
        // Validation accuracy as a percentage
        let validationAccuracy = (1.0 - trainingModel!.validationMetrics.classificationError) * 100
        print("validationAccuracy = \(validationAccuracy)")
        print("\n===== END TRAIN =====\n")
        
        print("\n===== START EVALUATE =====\n")
        let evaluationMetrics = trainingModel!.evaluation(on: testingData, textColumn: "SentimentText", labelColumn: "Sentiment")
        print("evaluationMetrics：")
        print(evaluationMetrics)
        let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
        print("evaluationAccuracy = \(evaluationAccuracy)")
        print("\n===== END EVALUATE =====\n")
    }
    
    private func saveModel() {
        guard let model = trainingModel else {
            fatalError("trainingModel is nil")
        }
        let metadata = MLModelMetadata(author: "Roger Wang", shortDescription: "This model analyzes the sentiment of a given IMDB review.", version: "1.0")
        
        print("save model...")
        do {
            try model.write(to: URL(fileURLWithPath: saveModelPath), metadata: metadata)
        } catch {
            print("err... writing model")
        }
    }
}
