//
//  ModelHandler.swift
//  Silero-VAD-for-iOS
//
//  Created by rajeev-tesseractDev on 05/16/2025.
//  Copyright (c) 2025 rajeev-tesseractDev. All rights reserved.
//

import Foundation
import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import onnxruntime_objc

/// Represents the result of a model prediction
struct Result {
    /// The processing time in milliseconds
    let processTimeMs: Double
    /// The output score from the model
    let score: Float
    /// The hidden state output from the LSTM layer
    let hn: ORTValue
    /// The cell state output from the LSTM layer
    let cn: ORTValue
}

/// Errors that can occur during model operations
enum OrtModelError: Error {
    case error(_ message: String)
}

/// A class for handling ONNX model operations including loading, running predictions, and managing LSTM states
final class ModelHandler: NSObject {
    // MARK: - Inference Properties
    
    /// The number of threads to use for inference
    let threadCount: Int32
    /// The maximum allowed number of threads
    let threadCountLimit = 10
    
    // MARK: - Model Parameters
    
    /// The batch size for model input
    let batchSize = 1
    /// The number of input channels
    let inputChannels = 3
    /// The width of the input tensor
    let inputWidth = 300
    /// The height of the input tensor
    let inputHeight = 300
    
    /// The dimension of each LSTM unit
    let lstm_unit_dimension = 64
    /// The number of LSTM units
    let lstm_unit_num = 2
    
    /// The last sample rate used
    var _last_sr: Int = 0
    /// The last batch size used
    var _last_batch_size: Int = 0
    
    /// The current hidden state of the LSTM layer
    var _h: ORTValue!
    /// The current cell state of the LSTM layer
    var _c: ORTValue!
    
    private var session: ORTSession
    private var env: ORTEnv
    
    /// Initializes the model handler with the specified model file
    /// - Parameters:
    ///   - modelFilename: The name of the model file
    ///   - modelExtension: The file extension of the model
    ///   - threadCount: The number of threads to use for inference (default: 1)
    init?(modelFilename: String, modelExtension: String, threadCount: Int32 = 1) {
        guard let modelPath = Bundle.main.url(forResource: modelFilename, withExtension: modelExtension)?.path() else {
            print("Failed to get model file path with name.")
            return nil
        }
        
        self.threadCount = threadCount
        do {
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.warning)
            try options.setIntraOpNumThreads(threadCount)
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession.")
            return nil
        }
        
        super.init()
    }
    
    /// Checks if the parameters have changed and resets state if necessary
    /// - Parameters:
    ///   - batchSize: The current batch size
    ///   - sr: The current sample rate
    func parameterCheck(batchSize: Int, sr: Int) {
        guard _last_batch_size == batchSize,
              _last_sr == sr else {
            resetState(batchSize: batchSize)
            return
        }
    }
    
    /// Parses an ORTValue to extract a Float value
    /// - Parameter value: The ORTValue to parse
    /// - Returns: The extracted Float value
    /// - Throws: OrtModelError if the value cannot be parsed
    func _parseToFloat(value: ORTValue?) throws -> Float {
        guard let rawOutputValue = value else {
            throw OrtModelError.error("failed to get model output")
        }
        let rawOutputData = try rawOutputValue.tensorData() as Data
        let floatValue = rawOutputData.withUnsafeBytes { $0.load(as: Float.self) }
        return floatValue
    }
    
    /// Runs a prediction with the given input tensors
    /// - Parameter inputTensors: An array of input tensors
    /// - Returns: A Result struct containing the prediction results
    /// - Throws: OrtModelError if the prediction fails
    func _prediction(inputTensors: [ORTValue]) throws -> Result {
        let inputNames = ["input", "sr", "h", "c"]
        let outputNames: Set<String> = ["output", "hn", "cn"]
        
        guard inputTensors.count == inputNames.count else {
            throw OrtModelError.error("inputTensors.count != inputNames.count")
        }
        
        let inputDic = Dictionary(uniqueKeysWithValues: zip(inputNames, inputTensors))
        
        let interval: TimeInterval
        let startDate = Date()
        let outputs:[String: ORTValue] = try session.run(withInputs: inputDic,
                                      outputNames: outputNames,
                                      runOptions: nil)
        interval = Date().timeIntervalSince(startDate) * 1000
        
        
        let score = try _parseToFloat(value: outputs["output"])
        
        guard let hn:ORTValue = outputs["hn"],
              let cn:ORTValue = outputs["cn"] else {
            throw OrtModelError.error("hn cn is not exist")
        }
        
        // Return ORT SessionRun result
        return Result(processTimeMs: interval, score: score, hn: hn, cn: cn)
    }
}

extension ModelHandler {
    /// Runs a prediction with the given input data and sample rate
    /// - Parameters:
    ///   - x: The input data as a Data object
    ///   - sr: The sample rate (8k or 16k)
    /// - Returns: The prediction score as a Float
    func prediction(x: Data, sr: Int64) -> Float {
        do {
            let size = x.count / MemoryLayout<Float>.size
            let inputShape: [NSNumber] = [batchSize as NSNumber,
                                          size as NSNumber]
            let xTensor:ORTValue = try ORTValue(tensorData: NSMutableData(data: x),
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape)
        
            let inputShape2: [NSNumber] = []
            let srData = withUnsafeBytes(of: sr) { Data($0) }
            let srTensor:ORTValue = try ORTValue(tensorData: NSMutableData(data: srData), elementType: .int64, shape: inputShape2)
            
            let inputTensors:[ORTValue] = [xTensor, srTensor, _h, _c]
            let predictionResult = try _prediction(inputTensors: inputTensors)
            
            _h = predictionResult.hn
            _c = predictionResult.cn
            return predictionResult.score
        } catch {
            print("Unknown error: \(error)")
        }
        
        return 0
    }
    
    /// Resets the LSTM states to zero
    /// - Parameter batchSize: The batch size to use for the reset states (default: 1)
    func resetState(batchSize: Int = 1) {
        _last_sr = 0
        _last_batch_size = 0
        let inputShape: [NSNumber] = [lstm_unit_num as NSNumber,
                                      batchSize as NSNumber,
                                      lstm_unit_dimension as NSNumber]
        
        let dataCount = inputShape.reduce(1, { $0 * ($1 as! Int) })
        let zeroData = Data(repeating: 0, count: dataCount * MemoryLayout<Float>.size)
        
        _h = try! ORTValue(tensorData: NSMutableData(data: zeroData),
                                    elementType: ORTTensorElementDataType.float,
                                    shape: inputShape)
        _c = try! ORTValue(tensorData: NSMutableData(data: zeroData),
                                    elementType: ORTTensorElementDataType.float,
                                    shape: inputShape)
    }
}
