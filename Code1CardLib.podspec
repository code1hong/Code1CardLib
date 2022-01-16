#
# Be sure to run `pod lib lint Code1CardLib.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  s.name             = 'Code1CardLib'
  s.version          = '0.1.0'
  s.summary          = 'Code1System Card Module.'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
TODO: Add long description of the pod here.
                       DESC

  s.homepage         = 'https://github.com/code1hong/Code1CardLib'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'code1hong' => 'code1hong@gmail.com' }
  s.source           = { :git => 'https://github.com/code1hong/Code1CardLib.git', :tag => s.version.to_s }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '10.0'

  s.source_files = 'Code1CardLib/Classes/**/*'
  
  s.swift_version = '5.0'
  
  s.static_framework = true
  s.dependency 'TensorFlowLiteSwift', '~> 2.3.0'
  s.dependency 'CryptoSwift', '~> 1.3.8'

  s.resources = ["Code1CardLib/res/card_s-fp16.tflite", "Code1CardLib/res/classes.txt", "Code1CardLib/res/Live.storyboard", "Code1CardLib/res/Code1License.lic"]

  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64'}
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64'}
  
  # s.resource_bundles = {
  #   'Code1CardLib' => ['Code1CardLib/Assets/*.png']
  # }

  # s.public_header_files = 'Pod/Classes/**/*.h'
  # s.frameworks = 'UIKit', 'MapKit'
  # s.dependency 'AFNetworking', '~> 2.3'
end
