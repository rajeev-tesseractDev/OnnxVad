//
//  VoiceActivityDetector.swift
//  Silero-VAD-for-iOS
//
//  Created by rajeev-tesseractDev on 05/16/2025.
//  Copyright (c) 2025 rajeev-tesseractDev. All rights reserved.
//

import Foundation
import AVFAudio
import onnxruntime_objc

/// Enum representing different voice activity detection modes
enum DetectMode {
    /// Process audio in complete chunks
    case Chunk
    /// Process audio in continuous streams with specified window size
    case Stream(windowSampleNums: Int)
}

/// Structure representing voice activity detection results
public struct VADResult {
    /// Confidence score of voice activity (0 to 1)
    public var score: Float
    /// Start sample index of the detected segment
    public var start: Int
    /// End sample index of the detected segment
    public var end: Int
}

/// Structure representing voice activity time segments
public struct VADTimeResult {
    /// Start time of voice activity (in samples)
    public var start: Int = 0
    /// End time of voice activity (in samples)
    public var end: Int = 0
}

extension Data {
    /// Converts Data to an array of Float values
    func floatArray() -> [Float] {
        var floatArray = [Float](repeating: 0, count: self.count/MemoryLayout<Float>.stride)
        _ = floatArray.withUnsafeMutableBytes {
            self.copyBytes(to: $0, from: 0..<count)
        }
        return floatArray
    }
    
    /// Enum representing byte ordering
    enum Endianess {
        case little
        case big
    }
    
    /// Converts Data to a Float value with specified byte ordering
    func toFloat(endianess: Endianess = .little) -> Float? {
        guard self.count <= 4 else { return nil }
        switch endianess {
        case .big:
            let data = [UInt8](repeating: 0x00, count: 4-self.count) + self
            return data.withUnsafeBytes { $0.load(as: Float.self) }
        case .little:
            let data = self + [UInt8](repeating: 0x00, count: 4-self.count)
            return data.reversed().withUnsafeBytes { $0.load(as: Float.self) }
        }
    }
}

/// A voice activity detector using Silero VAD model
public final class VoiceActivityDetector {
    private var _modelHandler: ModelHandler?
    private let expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)
    private var _detectMode: DetectMode = .Chunk
    
    /// Initializes a new voice activity detector
    public init() {
        loadModel()
    }
    
    /// Loads the Silero VAD ONNX model
    private func loadModel() {
        guard _modelHandler == nil else {
            return
        }
        
        _modelHandler = ModelHandler(modelFilename: "silero_vad_cache", modelExtension: "onnx", threadCount: 4)
    }
    
    /// Verifies if the audio format matches the expected format
    private func _checkAudioFormat(pcmFormat: AVAudioFormat) -> Bool {
        // Check whether the sample rate matches
        guard pcmFormat.sampleRate == expectedFormat!.sampleRate else {
            return false
        }
        
        // Check whether the number of channels matches
        guard pcmFormat.channelCount == expectedFormat!.channelCount else {
            return false
        }
        
        // Check if the bit depths match
        guard pcmFormat.commonFormat == expectedFormat!.commonFormat else {
            return false
        }
        
        return true
    }
    
    /// Divides a large sample count into smaller segments
    func divideIntoSegments(_ x: Int, step: Int) -> [(start: Int, count: Int)] {
        var result: [(start: Int, count: Int)] = []
        var remaining = x
        var start = 0
        
        while remaining > 0 {
            let count = min(step, remaining)
            result.append((start, count))
            remaining -= count
            start += count
        }
        
        return result
    }
    
    /// Performs voice activity detection on the audio buffer
    fileprivate func _detectVAD(_ buffer: AVAudioPCMBuffer, _ windowSampleNums: Int, _ modelHandler: ModelHandler ) -> [VADResult]  {
        var scores: [VADResult] = []
        let channelData: UnsafePointer<UnsafeMutablePointer<Float32>> = buffer.floatChannelData!
        let channelPointer: UnsafeMutablePointer<Float32> = channelData[0]
        let frameLength = Int(buffer.frameLength)
        
        let segments = divideIntoSegments(frameLength, step: windowSampleNums)
        
        var tempCount = 0
        segments.forEach { (start: Int, count: Int) in
            let pointer: UnsafeMutablePointer<Float32> = channelPointer.advanced(by: start)
            
            let byteSize = count * MemoryLayout<Float32>.stride
            var data = Data(bytes: pointer, count: byteSize)
            tempCount += count
            if count < windowSampleNums {
                data.append(Data(repeating: 0, count: windowSampleNums - count))
            }
            
            let score = modelHandler.prediction(x: data, sr: 16000)
            scores.append(VADResult(score: score, start: start, end: tempCount-1))
        }
        
        return scores
    }
}

public extension VoiceActivityDetector {
    /// Resets the internal state of the VAD model
    func resetState() {
        guard let modelHandler = _modelHandler else {
            return
        }
        _detectMode = .Chunk
        modelHandler.resetState()
    }
    
    /// Detects voice activity in an audio buffer (chunk mode)
    /// - Parameters:
    ///   - buffer: The audio buffer to analyze
    ///   - windowSampleNums: Number of samples per analysis window (default: 512)
    /// - Returns: Array of VAD results or nil if detection fails
    func detect(buffer: AVAudioPCMBuffer, windowSampleNums: Int = 512) -> [VADResult]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        guard _checkAudioFormat(pcmFormat: buffer.format) else {
            return nil
        }
        resetState()
        return _detectVAD(buffer, windowSampleNums, modelHandler)
    }
    
    /// Detects voice activity in continuous streaming mode
    /// - Parameters:
    ///   - buffer: The audio buffer to analyze
    ///   - windowSampleNums: Number of samples per analysis window (default: 512)
    /// - Returns: Array of VAD results or nil if detection fails
    func detectContinuously(buffer: AVAudioPCMBuffer, windowSampleNums: Int = 512) -> [VADResult]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        guard _checkAudioFormat(pcmFormat: buffer.format) else {
            return nil
        }
        
        switch _detectMode {
        case .Stream(windowSampleNums: windowSampleNums):
            break
        default:
            _detectMode = .Stream(windowSampleNums: windowSampleNums)
            modelHandler.resetState()
            break
        }
        
        return _detectVAD(buffer, windowSampleNums, modelHandler)
    }
    
    /**
     Detects voice activity with advanced timing parameters
     
     - Parameters:
        - buffer: Audio buffer to analyze
        - threshold: Speech probability threshold (0-1, default 0.5)
        - minSpeechDurationInMS: Minimum speech duration in milliseconds (default 250)
        - maxSpeechDurationInS: Maximum speech duration in seconds (default 30)
        - minSilenceDurationInMS: Minimum silence duration in milliseconds (default 100)
        - speechPadInMS: Padding duration in milliseconds (default 30)
        - windowSampleNums: Number of samples per analysis window (default 512)
     
     - Returns: Array of voice activity time segments or nil if detection fails
     */
    func detectForTimeStemp(buffer: AVAudioPCMBuffer,
                            threshold: Float = 0.5,
                            minSpeechDurationInMS: Int = 250,
                            maxSpeechDurationInS: Float = 30,
                            minSilenceDurationInMS: Int = 100,
                            speechPadInMS: Int = 30,
                            windowSampleNums: Int = 512) -> [VADTimeResult]? {
        
        let sr = buffer.format.sampleRate
        
        guard let vadResults = detect(buffer: buffer, windowSampleNums: windowSampleNums) else {
            return nil
        }
        
        let minSpeechSamples = Int(sr * Double(minSpeechDurationInMS) * 0.001)
        let maxSpeechSamples = Int(sr * Double(maxSpeechDurationInS))
        let minSilenceSample = Int(sr * Double(minSilenceDurationInMS) * 0.001)
        let minSilenceSampleAtMaxSpeech = Int(sr * Double(0.098))
        let speechPadSamples = Int(sr *  Double(speechPadInMS) * 0.001)
        
        var triggered = false
        var speeches = [VADTimeResult]()
        var currentSpeech = VADTimeResult()
        
        let neg_threshold = threshold - 0.15
        var temp_end = 0
        var prev_end = 0
        var next_start = 0
        
        for (i, speech) in vadResults.enumerated() {
            let speech_prob = speech.score
            if speech_prob >= threshold && temp_end != 0 {
                temp_end = 0
                if next_start < prev_end {
                    next_start = windowSampleNums * i
                }
            }
            
            
            if speech_prob >= threshold && !triggered {
                triggered = true
                currentSpeech.start = windowSampleNums * i
                continue
            }
            
            if triggered && (windowSampleNums * i) - currentSpeech.start > maxSpeechSamples {
                if prev_end != 0 {
                    currentSpeech.end = prev_end
                    speeches.append(currentSpeech)
                    currentSpeech = VADTimeResult()
                    if next_start < prev_end {
                        triggered = false
                    } else {
                        currentSpeech.start = next_start
                    }
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                } else {
                    currentSpeech.end = windowSampleNums * i
                    speeches.append(currentSpeech)
                    currentSpeech = VADTimeResult()
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = false
                    continue
                }
            }
            
            if speech_prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = windowSampleNums * i
                }
                if (windowSampleNums * i) - temp_end > minSilenceSampleAtMaxSpeech {
                    prev_end = temp_end
                }
                if (windowSampleNums * i) - temp_end < minSilenceSample {
                    continue
                } else {
                    currentSpeech.end = temp_end
                    if (currentSpeech.end - currentSpeech.start) > minSpeechSamples {
                        speeches.append(currentSpeech)
                    }
                    currentSpeech = VADTimeResult()
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = false
                    continue
                }
            }
        }
        
        let audio_length_samples = Int(buffer.frameLength)
        if currentSpeech.start > 0 && (audio_length_samples - currentSpeech.start) > minSpeechSamples {
            currentSpeech.end = audio_length_samples
            speeches.append(currentSpeech)
        }
        
        
        for i in 0..<speeches.count {
            if i == 0 {
                speeches[i].start = Int(max(0, speeches[i].start - speechPadSamples))
            }
            
            if i != speeches.count - 1 {
                let silence_duration = speeches[i+1].start - speeches[i].end
                if silence_duration < 2 * speechPadSamples {
                    speeches[i].end += Int(silence_duration / 2)
                    speeches[i+1].start = Int(max(0, speeches[i+1].start - silence_duration / 2))
                } else {
                    speeches[i].end = Int(min(audio_length_samples, speeches[i].end + speechPadSamples))
                    speeches[i+1].start = Int(max(0, speeches[i+1].start - speechPadSamples))
                }
            } else {
                speeches[i].end = Int(min(audio_length_samples, speeches[i].end + speechPadSamples))
            }
        }
        
        return speeches
    }
}
