// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate
import Photos


struct Card {
    var cardNumber: String!
    var cardMonth: String!
    var cardYear: String!
}

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferenceTime: Double
    let inferences: [Inference]
}

struct InferenceResult {
    var resultText: String
    var score: Float
}

/// Stores one formatted inference.
struct Inference {
    let confidence: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum MobileNetSSD {
//    static let modelInfo: FileInfo = (name: "passport_s-fp16", extension: "tflite")
//    static let labelsInfo: FileInfo = (name: "labelmap", extension: "txt")
    static let modelInfo: FileInfo = (name: "card_s-fp16", extension: "tflite")
    static let labelsInfo: FileInfo = (name: "classes", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10
    
    let threshold: Float = 0.25
    
    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640
    let inputHeight = 640
    
    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5
    
    // MARK: Private properties
    private var labels: [String] = []
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    // MARK: - Initialization
    
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 4) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = 4
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
        
        // Load the classes listed in the labels file.
        loadLabels(fileInfo: labelsFileInfo)
    }
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer, guideRect: CGRect, imageScale: CGFloat) -> Card? {
        
        let w: CGFloat!
        let h: CGFloat!
        let x: CGFloat!
        let y: CGFloat!
        
        if guideRect.width > guideRect.height {
            w = guideRect.width * imageScale
            h = guideRect.width * imageScale
            x = guideRect.minX * imageScale
            y = guideRect.minY * imageScale - (guideRect.width * imageScale - guideRect.height * imageScale) / 2
        }
        else {
            w = guideRect.height * imageScale
            h = guideRect.height * imageScale
            x = guideRect.minX * imageScale  - (guideRect.height * imageScale - guideRect.width * imageScale) / 2
            y = guideRect.minY * imageScale
        }
     
        // 테스트
        let newPixelBuffer = pixelBuffer.crop(to: CGRect(x: x, y: y, width: w, height: h))!
        
        let imageWidth = CVPixelBufferGetWidth(newPixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(newPixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(newPixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = newPixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
////        // 앨범 저장 코드
//        let uiImage = UIUtilities.createUIImage(from: scaledPixelBuffer, orientation: UIImage.Orientation.up)
//        PHPhotoLibrary.requestAuthorization { status in
//                    guard status == .authorized, let image = uiImage else { return }
//                    PHPhotoLibrary.shared().performChanges({
//                        PHAssetChangeRequest.creationRequestForAsset(from: image)
//                    }, completionHandler: nil)
//                }

        let interval: TimeInterval
        let outputBoundingBox: Tensor
//        let outputClasses: Tensor
//        let outputScores: Tensor
//        let outputCount: Tensor
        do {
            let inputTensor = try interpreter.input(at: 0)
    
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            interval = Date().timeIntervalSince(startDate) * 1000
            outputBoundingBox = try interpreter.output(at: 0)
//            outputClasses = try interpreter.output(at: 0)
//            outputScores = try interpreter.output(at: 0)
//            outputCount = try interpreter.output(at: 0)

        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Formats the results
//        let resultArray = formatResults(
//            boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
//            outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
//            outputScores: [Float](unsafeData: outputScores.data) ?? [],
//            outputCount: Int(([Float](unsafeData: outputCount.data) ?? [0])[0]),
//            width: CGFloat(imageWidth),
//            height: CGFloat(imageHeight)
//        )
        
        let outputs = ([Float](unsafeData: outputBoundingBox.data) ?? []) as [NSNumber]
        
        let nmsPredictions = PrePostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: 1.0, imgScaleY: 1.0, ivScaleX: 1.0, ivScaleY: 1.0, startX: 0.0, startY: 0.0)

        let cardResult = new_sortCard(predictions: nmsPredictions)
        
//        // Returns the inference time and inferences
//        let result = Result(inferenceTime: interval, inferences: resultArray)
        return cardResult
    }
    
    /// Filters out all the results with confidence score < threshold and returns the top N results
    /// sorted in descending order.
    func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
        var resultsArray: [Inference] = []
        if (outputCount == 0) {
            return resultsArray
        }
        for i in 0...outputCount - 1 {
            
            let score = outputScores[i]
            
            // Filters results with confidence < threshold.
            guard score >= threshold else {
                continue
            }
            
            // Gets the output class names for detected classes from labels list.
            let outputClassIndex = Int(outputClasses[i])
            let outputClass = labels[outputClassIndex + 1]
            
            var rect: CGRect = CGRect.zero
            
            // Translates the detected bounding box to CGRect.
            rect.origin.y = CGFloat(boundingBox[4*i])
            rect.origin.x = CGFloat(boundingBox[4*i+1])
            rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
            rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x
            
            // The detected corners are for model dimensions. So we scale the rect with respect to the
            // actual image dimensions.
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            
            // Gets the color assigned for the class
            let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
            let inference = Inference(confidence: score,
                                      className: outputClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.confidence  > second.confidence
        }
        
        return resultsArray
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                       "valid labels file and try again.")
        }
    }
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append((Float(bytes[i]) - imageMean) / imageStd)
        }
        return Data(copyingBufferOf: floats)
    }
    
    /// This assigns color for a particular class.
    private func colorForClass(withIndex index: Int) -> UIColor {
        
        // We have a set of colors and the depending upon a stride, it assigns variations to of the base
        // colors to each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
    
    private func new_sortCard(predictions: [Prediction]) -> Card! {
        var cardResult = Card()
        
        var index: Int = 0
        var score: Float = 0
        var resultText: String = ""
        var number: Prediction!
        var date: Prediction!
        var numberValue: String = ""
        var dateValue: String = ""
        
        let sortXPrediction = predictions.sorted{$0.rect.origin.x < $1.rect.origin.x}
        
        for cls in sortXPrediction {
            if cls.classIndex == 11 {
                number = cls
            }
            else if cls.classIndex == 12 {
                date = cls
            }
            else {continue}
        }
        
      
        // 검출 실패 시 코드 작성
        if number == nil || date == nil {
            print("영역 검출 실패")
            return nil
        }
        
        // 두줄 카드 번호 처리를 위한 Y정렬 후 X정렬
        var numberTemp = [Prediction]()
        
        for cls in sortXPrediction {
            if cls.rect.minY > number.rect.minY && cls.rect.maxY < number.rect.maxY && cls.rect.minX > number.rect.minX && cls.rect.maxX < number.rect.maxX {
                numberValue.append(labels[cls.classIndex])

                numberTemp.append(cls)
                
                score = score + cls.score
                index = index + 1
            }
            else if cls.rect.minY > date.rect.minY && cls.rect.maxY < date.rect.maxY && cls.rect.minX > date.rect.minX && cls.rect.maxX < date.rect.maxX {
                dateValue.append(labels[cls.classIndex])
                score = score + cls.score
                index = index + 1
            }
            else {continue}
        }
        

        // 검출 실패 시 코드 작성
        if numberValue.count != 16 || dateValue.count != 5 {
            print("박스 내 갯수 부족1")
            return nil
        }
        
        
        // 두줄 카드 번호 처리를 위한 Y정렬 후 X정렬
        var numberLine1 = [Prediction]()
        var numberLine2 = [Prediction]()
        var numberIndex: Int = 0
        
        
        let sortYNumber = numberTemp.sorted{$0.rect.origin.y < $1.rect.origin.y}
        for cls in sortYNumber {
            if numberIndex < 8{
                numberLine1.append(cls)
                numberIndex = numberIndex + 1
            }
            else{
                numberLine2.append(cls)
            }
        }
        numberLine1.sort{$0.rect.origin.x < $1.rect.origin.x}
        numberLine2.sort{$0.rect.origin.x < $1.rect.origin.x}
        var twoLineNumber: String = ""
        for n in numberLine1{
            twoLineNumber.append(labels[n.classIndex])
        }
        for n in numberLine2{
            twoLineNumber.append(labels[n.classIndex])
        }
        
        print(twoLineNumber)
        
        // 검출 실패 시 코드 작성
        if twoLineNumber.count != 16 {
            print("박스 내 갯수 부족2")
            return nil
        }
        
        // 결과 저장
        if (numberLine1[0].rect.maxY < numberLine2[0].rect.minY) && (numberLine1[3].rect.maxY < numberLine2[3].rect.minY) && (numberLine1[5].rect.maxY < numberLine2[5].rect.minY) {
            cardResult.cardNumber = twoLineNumber
        }
        else{
            cardResult.cardNumber = numberValue
        }
        
        cardResult.cardMonth = String(dateValue[...dateValue.index(dateValue.startIndex, offsetBy: 1)])
        cardResult.cardYear = String(dateValue[dateValue.index(dateValue.startIndex, offsetBy: 3)...])
    
        
        if !(luhnCheck(cardResult.cardNumber as String)) {
            print("카드번호 체크섬")
            return nil
        }
        
        if Int(cardResult.cardMonth)! > 12 || Int(cardResult.cardMonth)! < 1 {
            print("month 값 오류")
            return nil
        }
        if Int(cardResult.cardYear)! > 50 || Int(cardResult.cardYear)! < 10 {
            print("year 값 오류")
            return nil
        }
        
        var countNumber: Int = 1
        var numberResult: String = ""
        for s in numberValue{
            numberResult.append(s)
            if countNumber % 4 == 0 {numberResult.append(" ")}
            countNumber = countNumber + 1
        }
        
        resultText.append(numberResult)
        resultText.append("\n")
        resultText.append(dateValue)
        
        print(resultText)
        
        return cardResult
    }
    
    func luhnCheck(_ number: String) -> Bool {
        var sum = 0
        let digitStrings = number.reversed().map { String($0) }

        for tuple in digitStrings.enumerated() {
            if let digit = Int(tuple.element) {
                let odd = tuple.offset % 2 == 1

                switch (odd, digit) {
                case (true, 9):
                    sum += 9
                case (true, 0...8):
                    sum += (digit * 2) % 9
                default:
                    sum += digit
                }
            } else {
                return false
            }
        }
        return sum % 10 == 0
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}

