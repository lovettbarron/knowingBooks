#include "testApp.h"

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void testApp::setup(){
    
    panelWidth = 200; imgWidth = 640;
    scale = 1;
    panel.setup(panelWidth, 480);
    
    panel.addPanel("Settings");
    panel.addSlider("blur", 5, 1,127,true);
    panel.addToggle("thresholdOn",true);
    panel.addSlider("threshold",127,0,255,false);
    panel.addLabel("Canny Edge");
    panel.addSlider("CannyThresh1", 66.0, 0,255,false);
    panel.addSlider("CannyThresh2", 66.0, 0,255,false);
    
    panel.addLabel("HoughLines");
    panel.addSlider("HoughRho", 1, 1, 5, true);
    panel.addSlider("HoughTheta", 180, 0,360,false);
    panel.addSlider("HoughThresh", 50, 0,255, true);
    
    kinect.setRegistration(true);
    ofLog() << "Starting first kinect";
    kinect.init(false, true, true); // infrared=false, video=true, texture=true
    kinect.open(0);
    
    if(!kinect.isConnected()) {
        cam.initGrabber(640, 480);
    } else {
        img.create(480, 640, CV_8UC1);
    }

    imitate(thresh, img);
    imitate(threshImg, thresh);
    imitate(edgeImg,threshImg);
    imitate(intersectMat, thresh);
    imitate(intersectImg, intersectMat);
}

//--------------------------------------------------------------
void testApp::update(){
    if(!kinect.isConnected()) { cam.update(); } else { kinect.update(); }
	if(cam.isFrameNew() || kinect.isFrameNew()) {
        
        lines.clear(); intersect.clear(); poly.clear();
		if(!kinect.isConnected()) convertColor(cam, img, CV_RGB2GRAY); // Convert to grayscale
        else img = toCv(kinect.getDepthPixelsRef());
        
        cv::resize(img, img, cv::Size(0,0), 1/scale, 1/scale); // Resize for speed
        medianBlur(img, panel.getValueI("blur")); // Median blur instead of blur?
        
        if(panel.getValueB("thresholdOn")) threshold(img, thresh, panel.getValueI("threshold"));
        else thresh = img;
        
        float cThresh1 = panel.getValueF("CannyThresh1");
        float cThresh2 = panel.getValueF("CannyThresh2");
        int cAppSize = 3;
        
        cv::Canny(thresh, edges, cThresh1, cThresh2, cAppSize);
        
        double rho = panel.getValueI("HoughRho");
        double theta = CV_PI / panel.getValueI("HoughTheta");
        double hThresh = panel.getValueI("HoughThresh");
        
        cv::HoughLines( edges, lines, rho, theta, hThresh, 0, 0 );
        
        // This tries the approx poly method
        getIntersect( &lines, &intersect );
        
        getPoly( &edges, &poly );
        
        
        toOf(thresh, threshImg);
        threshImg.update();
        
        toOf(edges, edgeImg);
        edgeImg.update();
        
        toOf(houghMat, houghImg);
        houghImg.update();
        
        toOf(intersectMat, intersectImg);
        intersectImg.update();
	}
}

//--------------------------------------------------------------
void testApp::draw(){
    ofSetColor(255);
    ofPushMatrix();
    ofTranslate(panelWidth,0);
    if(!kinect.isConnected()) cam.draw(0, 0);
    else kinect.draw(0,0);
    
    ofVec2f debugSize( img.cols*scale/2, img.rows*scale/2 );
    
    threshImg.draw(imgWidth, 0, debugSize.x, debugSize.y);
    edgeImg.draw(imgWidth,debugSize.y, debugSize.x, debugSize.y);
    intersectImg.draw(imgWidth,debugSize.y * 2,debugSize.x, debugSize.y);
    houghImg.draw(imgWidth-debugSize.x,debugSize.y * 2,debugSize.x, debugSize.y);
    
    // Draw the houghLine approach
    if(intersect.size() > 0) {
        for(int i=0;i<intersect.size();i++) {
            ofPoint inters = toOf(intersect.at(i))*scale;
            ofSetColor(255,0,0);
            ofCircle(inters,10);
            ofSetColor(255);
            ofDrawBitmapString(ofToString(i), inters);
        }
    }
    
    // Draw the findContour approach
    if( poly.size() > 0) {
        for( int i=0;i<poly.size();i++) {
            for(int j=1; j<poly[i].size(); j++) {
                ofPoint coord = toOf(poly[i][j]) * scale;
                ofPoint prevCoord = toOf(poly[i][j-1]) * scale;
                ofSetColor(0,255,255);
                ofLine(prevCoord, coord);
                if(j==1) ofCircle(prevCoord,5);
                ofCircle(coord,5);
                
                ofPoint center = toOf(getCenter(&poly[i]));
                ofDrawBitmapString(ofToString(i), center);
            }
            
        }
    }
    ofPopMatrix();
    glDisable(GL_DEPTH_TEST);
}

//--------------------------------------------------------------
//                  Get Contour approach
//--------------------------------------------------------------
void testApp::getPoly( cv::Mat * src, vector< vector<cv::Point> > * dst ) {
    dst->clear();
    vector< vector<cv::Point> > contours;
    vector<cv::Point> approx;
    dilate(*src, *src, Mat(), cv::Point(-1,-1));
    findContours( *src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );
    approx.reserve(contours.size());
    
    for(int i=0;i<contours.size();i++) {
        double arc = arcLength( contours[i] , false)*0.02;
        cv::approxPolyDP(Mat(contours[i]), approx, arc, true);
        
        if (approx.size() == 4 &&
            fabs(contourArea(Mat(approx))) > 200 &&
            isContourConvex(Mat(approx)))
        {
            double maxCosine = 0;
            
            for (int j = 2; j < 5; j++)
            {
                double cosine = fabs( angle( approx[j%4], approx[j-2], approx[j-1] ) );
                maxCosine = MAX(maxCosine, cosine);
            }
            
            if (maxCosine < 0.3)
                dst->push_back(approx);
        }
    }
    
    
}

double testApp::angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

cv::Point testApp::getCenter( vector<cv::Point> * src ) {
    cv::Point cur, prev, centroid;
    double area = 0;
    for(int i=1;i<src->size();i++) {
        cur = src->at(i);
        prev = src->at(i-1);
        
        double a = cur.x * prev.y - prev.x * cur.y;
        area += a;
        centroid.x += (prev.x + cur.x)*a;
        centroid.y += (prev.y + cur.y)*a;
    }
    area * 0.5;
    centroid.x = centroid.x / (6.0*area);
    centroid.y = centroid.y / (6.0*area);
    return centroid;
}




//--------------------------------------------------------------
//                      Hough line stuff
//--------------------------------------------------------------
void testApp::getIntersect(vector<cv::Vec2f> * src, vector<cv::Vec2f> * dst) {
    intersectMat = cv::Mat::zeros(intersectMat.rows, intersectMat.cols, intersectMat.type());
    
    for( size_t i = 0; i < src->size(); i++ )
    {
        for(size_t j = 0; j < src->size(); j++)
        {
            cv::Vec2f line1 = src->at(i);
            cv::Vec2f line2 = src->at(j);
            if(acceptLinePair(line1, line2, CV_PI / 32))
            {
                cv::Vec2f intersection = computeIntersect(line1, line2);
                if( intersection[0] < intersectMat.cols * 1/scale && intersection[0] >= 0 
                   && intersection[1] < intersectMat.rows * 1/scale && intersection[1] >= 0) {
                    dst->push_back(intersection);
                    intersectMat.at<double>(intersection[0], intersection[1]) = 255;
                }
            }
        }
        
    }
    
   // intersectMat = Mat(*dst,);
    ofLog() << "Found " << dst->size() << " intersects";
}




//--------------------------------------------------------------
bool testApp::acceptLinePair(Vec2f line1, Vec2f line2, float minTheta) {
    float theta1 = line1[1], theta2 = line2[1];
    
    if(theta1 < minTheta)
    {
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }
    
    if(theta2 < minTheta)
    {
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }
    
    return abs(theta1 - theta2) > minTheta;
}

//--------------------------------------------------------------
Point2f testApp::computeIntersect(Vec2f line1, Vec2f line2) {
    vector<Point2f> p1 = lineToPointPair(line1);
    vector<Point2f> p2 = lineToPointPair(line2);
    
    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);
    
    return intersect;
}

//--------------------------------------------------------------
vector<Point2f> testApp::lineToPointPair(Vec2f line) {
    vector<Point2f> points;
    
    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;
    
    points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));
    
    return points;
}