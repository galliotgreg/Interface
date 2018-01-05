using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;
using Emgu.CV.Structure;
using Emgu.CV.Util;

public class Exemple : MonoBehaviour {

    public int widthBlur = 0;
    public int heightBlur = 0;
    public Size blurS;

    public int kSize = 17;

    public int widthGaussianBlur = 0;
    public int heightGaussianBlur = 0;
    public Size GaussianBlurS;
    public int sigmaX = 0;
    public int sigmaY = 0;

    public double MinH = 173.8;
    public double MaxH = 179.3;
    public double MinS = 149.4;
    public double MaxS = 255;
    public double MinV = 121.3;
    public double MaxV = 255;


    Hsv seuilBas;
    Hsv seuilHaut;

    Mat image;
    Mat imageBlur;
    Mat imageMedianBlur;
    Mat imageGaussianBlur;

    Mat elemntStruct;

    VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
    VectorOfPoint biggestContours;
    int biggestContoursIndex;
    double biggestContourArea;
    double biggestContourAreaOld = 0;

    bool isFiring;

    public AudioClip hadoken;
    AudioSource audio;

    public GameObject particle;

    VideoCapture webcam;
    //VideoWriter writer;

	// Use this for initialization
	void Start () {
        webcam = new VideoCapture(0);
        isFiring = false;
        audio = GetComponent<AudioSource>();          
        
        //webcam = new VideoCapture("D:\\Gamagora\\Interface\\Interface\\Assets\\ST.mkv");

        //testé avec .avi mais des erreurs lors de la sauvegarde
        //writer = new VideoWriter("D:/test.mp4", 30, new Size(300,300),true);
    }
	
	// Update is called once per frame
	void Update () {

        image = webcam.QueryFrame();
        imageBlur = webcam.QueryFrame();
        imageMedianBlur = webcam.QueryFrame();
        imageGaussianBlur = webcam.QueryFrame();

        elemntStruct = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3,3), new Point(1,1));

        seuilBas = new Hsv(MinH, MinS, MinV);
        seuilHaut = new Hsv(MaxH, MaxS, MaxV);

        // Miroir de l'image
        CvInvoke.Flip(image,image,FlipType.Horizontal);
        CvInvoke.Imshow("", image);
        //CvInvoke.CvtColor(image, image, ColorConversion.Bgr2Gray);
        CvInvoke.CvtColor(image, image, ColorConversion.Bgr2Hsv);
        CvInvoke.MedianBlur(image, image, kSize);
        Image<Hsv, byte> imageHsv = image.ToImage<Hsv, byte>();
        CvInvoke.Imshow("Seuil", imageHsv);
        Mat imGray = imageHsv.InRange(seuilBas, seuilHaut).Mat;

        
        CvInvoke.Erode(imGray, imGray, elemntStruct, new Point(1, 1), 3, BorderType.Default, new MCvScalar());
        CvInvoke.Dilate(imGray, imGray, elemntStruct, new Point(1, 1), 3, BorderType.Default, new MCvScalar());

        CvInvoke.FindContours(imGray, contours, null, RetrType.List, ChainApproxMethod.ChainApproxNone);

        biggestContourArea = 0;
        biggestContoursIndex = 0;
        for (int i=0; i < contours.Size; i++)
        {
            if (CvInvoke.ContourArea(contours[i]) > biggestContourArea)
            {
                biggestContours = contours[i];
                biggestContoursIndex = i;
                biggestContourArea = CvInvoke.ContourArea(contours[i]);
            }
        }
        
        CvInvoke.DrawContours(imGray, contours, biggestContoursIndex, new MCvScalar(150, 150, 0), 5);
        CvInvoke.DrawContours(imageHsv, contours, biggestContoursIndex, new MCvScalar(150, 0, 0), 5);
        CvInvoke.Imshow("Seuil", imGray);
        
        //if ((biggestContourAreaOld != 0 ) && (biggestContourArea > (biggestContourAreaOld - (biggestContourArea/2))))
        if(!isFiring && biggestContourArea > 30000)
        {
            isFiring = true;
            Debug.Log("Boule de Feu");
            audio.Play();
            GameObject clone = Instantiate(particle, transform.position, transform.rotation);            
            //clone.transform.position += clone.transform.forward * 1.0f * Time.deltaTime;
            Debug.Log(clone.transform.position);

        } else if (biggestContourArea < 3000)
        {
            isFiring = false;
        }

        biggestContourAreaOld = biggestContourArea;


        /*CvInvoke.Blur(image, imageBlur, new Size(5, 5), new Point(1,1), BorderType.Default);
        CvInvoke.MedianBlur(image, imageMedianBlur, 17);
        CvInvoke.GaussianBlur(image, imageGaussianBlur, new Size(5, 5), 31,0, BorderType.Default);

        CvInvoke.Imshow("Mon image Blur", imageBlur);
        CvInvoke.Imshow("Mon image Median", imageMedianBlur);
        CvInvoke.Imshow("Mon image Gaussian", imageGaussianBlur);*/
        CvInvoke.Imshow("Mon image", imageHsv);

        CvInvoke.Resize(image, image, new Size(300,300));
        //writer.Write(image);
	}

    private void OnDestroy()
    {
        CvInvoke.DestroyAllWindows();
    }
}
