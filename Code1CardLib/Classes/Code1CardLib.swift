
import Foundation

public class Code1CardLib: UIView {
    
    public func start(father: UIViewController) {
        
        
        // 라이센스 불러오기
        let filePath = Bundle.main.path(forResource: "Code1License", ofType: "lic")
        let license = try? String(contentsOfFile: filePath!).replacingOccurrences(of: "\n", with: "")

        // 번들아이디 체크
        let bundle = Bundle(for: Code1CardLib.self).bundleIdentifier

        //복호화
        let dec = AES128Util.decrypt(encoded: license!)

//        print(AES128Util.encrypt(string: bundle!))

        // 현재는 print 문이지만 추후에 고객에 맞춰 따라 라이센스 처리 코드 작성
        if dec == bundle {print("라이센스 체크 성공")}
        else {
            print("라이센스가 유효하지 않습니다.")
        }
        
        let storyboard = UIStoryboard(name: "Live", bundle: Bundle(for: Code1CardLib.self))
        if let vc = storyboard.instantiateViewController(withIdentifier: "View") as? InferenceController {
            father.navigationController?.pushViewController(vc,animated: true)
        }
    }

}


