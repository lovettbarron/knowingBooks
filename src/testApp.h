#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxKinect.h"
#include "ofxAutoControlPanel.h"

// From this stack overflow question about square detection and hough based detection
// http://stackoverflow.com/questions/10533233/opencv-c-obj-c-advanced-square-detection

// Also attempting to compare to the findContour approach described here
// http://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection


class testApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
    
        void getIntersect(vector<cv::Vec2f> * src, vector<cv::Vec2f> * dst);
    
//        void getPoly( vector<cv::Vec2f> * src, vector<cv::Vec2f> * dst);
        void getPoly( cv::Mat * src, vector< vector<cv::Point> > * dst );

        double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 );
        cv::Point getCenter( vector<cv::Point> * src );
    
        cv::Point2f computeIntersect(cv::Vec2f line1, cv::Vec2f line2);
        vector<cv::Point2f> lineToPointPair(cv::Vec2f line);
        bool acceptLinePair(cv::Vec2f line1, cv::Vec2f line2, float minTheta);
    
        int panelWidth, imgWidth, scale;
        ofxAutoControlPanel panel;
    
        ofVideoGrabber cam;
        ofxKinect kinect;
    
        cv::Mat img;
        cv::Mat thresh;
        cv::Mat edges;
        cv::Mat intersectMat;
        cv::Mat houghMat;
        cv::Mat contourMat;
    
        ofImage threshImg;
        ofImage edgeImg;
        ofImage intersectImg;
        ofImage houghImg;
    
        vector<cv::Vec2f> lines;
        vector<cv::Vec2f> intersect;
        vector< vector<cv::Point> > poly;
    
};
