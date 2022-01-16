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

import UIKit

class InferenceController: UIViewController {
    
    // MARK: Storyboards Connections
    @IBOutlet weak var previewView: PreviewView!
    @IBOutlet weak var overlayView: OverlayView!
    @IBOutlet weak var resumeButton: UIButton!
    @IBOutlet weak var cameraUnavailableLabel: UILabel!
    
    @IBOutlet weak var bottomSheetStateImageView: UIImageView!
    @IBOutlet weak var bottomSheetView: UIView!
    @IBOutlet weak var bottomSheetViewBottomSpace: NSLayoutConstraint!
    
    // MARK: Constants
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    private let edgeOffset: CGFloat = 2.0
    private let labelOffset: CGFloat = 10.0
    private let animationDuration = 0.5
    private let collapseTransitionThreshold: CGFloat = -30.0
    private let expandTransitionThreshold: CGFloat = 30.0
    private let delayBetweenInferencesMs: Double = 200
    
    // MARK: Instance Variables
    private var initialBottomSpace: CGFloat = 0.0
    
    // Holds the results at any time
    private var result: Result?
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    
    // MARK: Controllers that manage functionality
    private lazy var cameraFeedManager = CameraFeedManager(previewView: previewView)
    private var modelDataHandler: ModelDataHandler? =
    ModelDataHandler(modelFileInfo: MobileNetSSD.modelInfo, labelsFileInfo: MobileNetSSD.labelsInfo)

    
    // 추가
    var cardResult: Card!
    var finish = false
    var naviY: CGFloat!

    private var guideRect: CGRect!
    private var imageScaleRatio: CGFloat!
    private var verticalMode = true
    @IBOutlet weak var modeChangeBtn: UIButton!
    @IBOutlet weak var changeButton: UIButton!
    @IBOutlet weak var guideLabel: UILabel!
    
    // MARK: View Handling Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        
        naviY = navigationController?.navigationBar.frame.maxY
        
        previewView.frame = CGRect(x: 0, y: (self.navigationController?.navigationBar.frame.maxY)!, width: self.view.frame.width, height: self.view.frame.width * (4032 / 3024))
        
        // 촬영 모드 설정 및 버튼 적용
        setupCardGuide(mode: "horizontal")
        modeChangeBtn.setTitle("세로카드 스캔", for: .normal)
        modeChangeBtn.setImage(#imageLiteral(resourceName: "bt_ic_card"), for: .normal)
        modeChangeBtn.setTitleColor(.black, for: .normal)
        modeChangeBtn.imageView?.contentMode = .scaleAspectFit
        modeChangeBtn.titleLabel?.font = .boldSystemFont(ofSize: 18)
        modeChangeBtn.contentHorizontalAlignment = .center
        modeChangeBtn.semanticContentAttribute = .forceLeftToRight
        modeChangeBtn.imageEdgeInsets = .init(top: 0, left: 0, bottom: 0, right: 0)
        
        guard modelDataHandler != nil else {
            fatalError("Failed to load model")
        }
        cameraFeedManager.delegate = self
        
        imageScaleRatio = CGFloat(1080) / previewView.frame.width
//        overlayView.clearsContextBeforeDrawing = true
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.finish = false
        self.cardResult = nil
//        changeBottomViewState()
        cameraFeedManager.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraFeedManager.stopSession()
    }
    
//    override func viewDidAppear(_ animated: Bool) {
//        super.viewDidAppear(animated)
//    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    // MARK: Button Actions
    @IBAction func onClickResumeButton(_ sender: Any) {
        
        cameraFeedManager.resumeInterruptedSession { (complete) in
            
            if complete {
                self.resumeButton.isHidden = true
                self.cameraUnavailableLabel.isHidden = true
            }
            else {
                self.presentUnableToResumeSessionAlert()
            }
        }
    }
    
    func presentUnableToResumeSessionAlert() {
        let alert = UIAlertController(
            title: "Unable to Resume Session",
            message: "There was an error while attempting to resume session.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        self.present(alert, animated: true)
    }

    // MARK: Storyboard Segue Handlers
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        
        if segue.identifier == "showResult" {
            if segue.destination is ResultController{
                let newController = segue.destination as? ResultController
                newController?.cardYear = cardResult.cardYear
                newController?.cardMonth = cardResult.cardMonth
                newController?.cardNumber = cardResult.cardNumber
            }
        }
        
    }
    
    // MARK: UI
    // 가이드 세팅
    // 세로모드 카드 가이드 설정
    func setupCardGuide(mode: String) {
        //뒷배경 색깔 및 투명도
        let maskLayerColor: UIColor = UIColor.white
        let maskLayerAlpha: CGFloat = 1.0
        var cardBoxLocationX: CGFloat!
        var cardBoxLocationY: CGFloat!
        var cardBoxWidthSize: CGFloat!
        var cardBoxheightSize: CGFloat!
        
        ////////////// 영역 설정
        if mode == "horizontal" {
            // 카드 가이드 박스 가로 사이즈 = 전체 영역 94%
            cardBoxWidthSize = view.bounds.width * 0.94
            // 카드 가이드 박스 세로 사이즈 = 전체 영역 40%
            cardBoxheightSize = cardBoxWidthSize / 1.58
            // 카드 가이드 박스 시작 X 좌표 = 전체 뷰 영역의 3% 위치
            cardBoxLocationX = view.bounds.width * 0.03
            // 카드 가이드 박스 시작 Y 좌표 = 전체 뷰 영역의 25% 위치
            cardBoxLocationY = view.bounds.height * 0.25
        }
        else {
            // 카드 가이드 박스 세로 사이즈 = 전체 영역 40%
            cardBoxheightSize = modeChangeBtn.frame.minY - guideLabel.frame.maxY - 60
            // 카드 가이드 박스 가로 사이즈 = 전체 영역 94%
            cardBoxWidthSize = cardBoxheightSize / 1.58
            // 카드 가이드 박스 시작 X 좌표 = 전체 뷰 영역의 3% 위치
            cardBoxLocationX = (view.bounds.width - cardBoxWidthSize) / 2
            // 카드 가이드 박스 시작 Y 좌표 = 전체 뷰 영역의 25% 위치
            cardBoxLocationY = view.bounds.height * 0.25
        }
        
        let cardRect = CGRect(x: cardBoxLocationX,
                                y: cardBoxLocationY,
                                width: cardBoxWidthSize,
                                height: cardBoxheightSize)

        guideRect = cardRect
        
        // 카드 가이드 백그라운드 설정
        let backLayer = CALayer()
        backLayer.frame = view.bounds
        backLayer.backgroundColor = maskLayerColor.withAlphaComponent(maskLayerAlpha).cgColor

        // 카드 가이드 구역 설정
        let maskLayer = CAShapeLayer()
        let path = UIBezierPath(roundedRect: cardRect, cornerRadius: 10.0)
        path.append(UIBezierPath(rect: view.bounds))
        maskLayer.path = path.cgPath
        maskLayer.fillRule = CAShapeLayerFillRule.evenOdd
        backLayer.mask = maskLayer
        self.view.layer.addSublayer(backLayer)
        
        self.view.bringSubviewToFront(modeChangeBtn)
        self.view.bringSubviewToFront(guideLabel)
    }
    @IBAction func changeModeButton(_ sender: Any) {
        self.view.layer.sublayers?.remove(at: self.view.layer.sublayers!.count - 1)
        // 세로로 변경
        if verticalMode == true {
            verticalMode = false
            setupCardGuide(mode: "vertical")
            changeButton.setTitle("가로전환", for: .normal)
            
        }
        // 가로로 변경
        else {
            verticalMode = true
            setupCardGuide(mode: "horizontal")
            changeButton.setTitle("세로전환", for: .normal)
    
        }
    }
    
    // 가로 세로 모드 변경
    @IBAction func changeMode(_ sender: Any) {
        overlayView.layer.sublayers?.remove(at: overlayView.layer.sublayers!.count - 3)
        
        // 세로로 변경
        if verticalMode == true {
            verticalMode = false
            setupCardGuide(mode: "vertical")
            modeChangeBtn.setTitle("가로카드 스캔", for: .normal)
            
        }
        // 가로로 변경
        else {
            verticalMode = true
            setupCardGuide(mode: "horizontal")
            modeChangeBtn.setTitle("세로카드 스캔", for: .normal)
    
        }
    }
    
}

// MARK: CameraFeedManagerDelegate Methods
extension InferenceController: CameraFeedManagerDelegate {
    
    func didOutput(pixelBuffer: CVPixelBuffer) {
        runModel(onPixelBuffer: pixelBuffer)
    }
    
    // MARK: Session Handling Alerts
    func sessionRunTimeErrorOccurred() {
        
        // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
        self.resumeButton.isHidden = false
    }
    
    func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
        
        // Updates the UI when session is interrupted.
        if resumeManually {
            self.resumeButton.isHidden = false
        }
        else {
            self.cameraUnavailableLabel.isHidden = false
        }
    }
    
    func sessionInterruptionEnded() {
        
        // Updates UI once session interruption has ended.
        if !self.cameraUnavailableLabel.isHidden {
            self.cameraUnavailableLabel.isHidden = true
        }
        
        if !self.resumeButton.isHidden {
            self.resumeButton.isHidden = true
        }
    }
    
    func presentVideoConfigurationErrorAlert() {
        
        let alertController = UIAlertController(title: "Configuration Failed", message: "Configuration of camera has failed.", preferredStyle: .alert)
        let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
        alertController.addAction(okAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentCameraPermissionsDeniedAlert() {
        
        let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
            
            UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
        }
        
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
        
    }
    
    /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    @objc func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        
        // Run the live camera pixelBuffer through tensorFlow to get the result
        
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        
        guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
            return
        }
        
        previousInferenceTimeMs = currentTimeMs
        cardResult = (self.modelDataHandler?.runModel(onFrame: pixelBuffer, guideRect: self.guideRect, imageScale: self.imageScaleRatio))
        
        
        DispatchQueue.main.async {
            if self.cardResult != nil && self.finish == false {
                self.finish = true
                self.performSegue(withIdentifier: "showResult", sender: self)
            }
        }
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            
            // Translates bounding box rect to current view.
            var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / imageSize.width, y: self.overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = self.edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = self.edgeOffset
            }
            
            if convertedRect.maxY > self.overlayView.bounds.maxY {
                convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            }
            
            if convertedRect.maxX > self.overlayView.bounds.maxX {
                convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            }
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(inference.className)  (\(confidenceValue)%)"
            
            let size = string.size(usingFont: self.displayFont)
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        
    }
    
    /** Calls methods to update overlay view with detected bounding boxes and class names.
     */
    func draw(objectOverlays: [ObjectOverlay]) {
        
        self.overlayView.objectOverlays = objectOverlays
        self.overlayView.setNeedsDisplay()
    }
    
}
