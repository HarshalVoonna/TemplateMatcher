import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class TemplateMatch {

	public static void run(String inFile, String templateFile, String outFile, int match_method) {
		System.out.println("\nRunning Template Matching");
		Mat img = Highgui.imread(inFile);
		Mat templ = Highgui.imread(templateFile);
		int result_cols = img.cols() - templ.cols() + 1;
		int result_rows = img.rows() - templ.rows() + 1;
		Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);
		Imgproc.matchTemplate(img, templ, result, match_method);
		Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());
		MinMaxLocResult mmr = Core.minMaxLoc(result);
		Point matchLoc;
		if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
			matchLoc = mmr.minLoc;
		} else {
			matchLoc = mmr.maxLoc;
		}
		Core.rectangle(img, matchLoc, new Point(matchLoc.x + templ.cols(),
				matchLoc.y + templ.rows()), new Scalar(0, 0, 0));
		System.out.println("Writing "+ outFile);
		Highgui.imwrite(outFile, img);
	}

	public static void main(String[] args){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME );
        run("pics/capturedImage.jpg","pics/templateImage.jpg","pics/outputImage.jpg", Imgproc.TM_CCOEFF);
	}
}
