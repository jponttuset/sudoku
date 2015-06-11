/* Basic includes */
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <sstream>

/* STL containers */
#include <set>
#include <vector>
#include <unordered_set>
#include <string>

/* OpenCV includes */
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>

/* Tesseract OCR includes */
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

/* Our functions to solve the Sudoku */
#include "sudoku.hpp"

using namespace cv;
using namespace std;

/* Canny parameters */
int ratio2       = 3;
int kernel_size  = 3;
int lowThreshold = 30;

/* Mode of painting, to illustrate the process
 *  0 - Just the image
 *  1 - Canny
 *  2 - Hough lines
 *  3 - Grid intersections
 *  4 - Number cells
 *  5 - Recognized numbers
 *  6 - Solved Sudoku
 */
int mode = 0;

/* Finds the intersection of two lines, or returns false.
 *  - The lines are defined by (o1, p1) and (o2, p2).
 *  - The intersection point is returned in 'inters'
 */
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                  Point2f &inters)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;
    
    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;
    
    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    inters = o1 + d1 * t1;
    return true;
}

/* Displays a point in the image (small circle) */
void paint_point( Mat& img, Point center )
{
    int thickness = -1;
    int lineType = 8;
    circle(img, center, 3, Scalar(0, 0, 255), thickness, lineType);
}

/* Struct that helps us store and sort pairs of lines */
struct PairStruct
{
    PairStruct(size_t new_id1=0, size_t new_id2=0, double new_inters=0)
    : id1(new_id1), id2(new_id2), inters(new_inters)
    {
        
    }
    size_t id1;
    size_t id2;
    double inters;
};

struct Line
{
    /* Extreme points */
    Point e1;
    Point e2;
    
    /* Id in the global container */
    std::size_t id;
    
    /* Id in the horizontal or vertical container */
    std::size_t hv_id;
    
    /* Intersections with orthogonal lines */
    multimap<double,size_t> intersections;
};

/* Function that takes two sets of lines, and looks for a pattern of ten evenly-distributed lines at the first set,
 * with respect to the intersections with the other set of lines.
 * - lines1 and lines2 are input vectors containing pairs with the 'Line' struct defining the lines, and their distance to the origin
 * - sel_lines is the output vector containing the ids of the ten 'recognized' sets of lines
 *   (e.g., all ids in sel_lines[0] are part of the left-most or upper line of the grid)
 * - The function returns true only if it finds an acceptable pattern of ten lines
 */
bool classify_lines(const vector<pair<double,Line>>& lines1, const vector<pair<double,Line>>& lines2, vector<set<size_t>>& sel_lines)
{
    double dist_th      = 20; /* At least 20 pixels apart between lines of different sets (coming from different *true* lines) */
    // double dist_th_down = 5;  /* At most 5 pixels apart within set */
    
    /* No lines? Nothing to do! */
    if (lines1.empty())
        return false;
    else
    {
        /* We store the distance between consecutive lines, to look for nine similar 'distances' */
        vector<pair<double,PairStruct>> int_diffs;
         
        /* Get the line in the middle of the detected lines (for this not to be one of the Sudoku lines,
         * there should be at least ten detected lines outside the grid... we hope the background is not that messy :)
         */
        size_t horiz_id = round(lines1.size()/2);
        auto line_it = lines1.begin()+horiz_id;
        const Line& middle_line = line_it->second;
        
        /* The line in the middle intersects with less than 10 lines, no Sudoku grid in the image */
        if (middle_line.intersections.size()<=9)
            return false;
        else
        {
            /* Scan all pairs of consecutive intersections with the middle line and store the 'pair'*/
            auto prev_inter = middle_line.intersections.begin();
            auto      inter = middle_line.intersections.begin();
            ++inter;
            for(; inter!=middle_line.intersections.end(); ++inter,++prev_inter)
                int_diffs.push_back(make_pair(inter->first-prev_inter->first,PairStruct(prev_inter->second, inter->second, inter->first)));
            
            /* Sort the pairs of consecutive intersections with respect to their distance */
            sort(int_diffs.begin(),int_diffs.end(),[](const std::pair<double,PairStruct> &left, const std::pair<double,PairStruct> &right) {return left.first < right.first;});
            
            /* Look for the round of 9 most similar differences */
            auto it1 = int_diffs.begin();
            auto it2 = int_diffs.begin()+8;
            double min_diff = 1000000;
            int min_ind = -1;
            size_t curr_ind = 0;
            for(;it2<int_diffs.end(); ++it1, ++it2, ++curr_ind)
            {
                if(it1->first>dist_th)
                {
                    if(it2->first-it1->first<min_diff)
                    {
                        min_diff = it2->first-it1->first;
                        min_ind  = (int)curr_ind;
                    }
                }
            }
            
            /* Have we found a 'round'? */
            if(min_ind<0)
                return false;
            else if(max(int_diffs[min_ind].first,int_diffs[min_ind+8].first)/min(int_diffs[min_ind].first,int_diffs[min_ind+8].first) > 1.3)
                return false;
            else
            {
                /* Put them together to sort them */
                vector<PairStruct> sel_pairs(9);
                for(std::size_t ii=0; ii<9; ++ii)
                    sel_pairs[ii] = int_diffs[min_ind+ii].second;
                sort(sel_pairs.begin(),sel_pairs.end(),[](const PairStruct &left, const PairStruct &right) {return left.inters < right.inters;});
                
                /* Start the sets of similar lines */
                sel_lines.resize(10);
                for(std::size_t ii=0; ii<9; ++ii)
                {
                    sel_lines[ii  ].insert(sel_pairs[ii].id1);
                    sel_lines[ii+1].insert(sel_pairs[ii].id2);
                }
                
//                /* Add the rest of the lines to the corresponding sets to do the mean */
//                for(auto inter:base_line.intersections)
//                    for(std::size_t ii=0; ii<10; ++ii)
//                        if(abs(inter.first-xxx))<dist_th_down)    // (*sel_lines[ii].begin()
//                            sel_lines[ii].insert(inter.second);
            }
        }
    }

    return true;
}


Point2f mean_intersection(const set<size_t>& h_set, const set<size_t>& v_set, const vector<pair<double,Line>>& horiz, const vector<pair<double,Line>>& verti)
{
    /* Get all interesections */
    std::vector<Point2f> all_int;
    for(auto it1:h_set)
    {
        for(auto it2:v_set)
        {
            Point2f inters;
            if(intersection(horiz[it1].second.e1, horiz[it1].second.e2,
                         verti[it2].second.e1, verti[it2].second.e2,
                         inters))
                all_int.push_back(inters);
        }
    }

    /* Get the mean */
    Point2f mean = all_int[0];
    for (std::size_t ii=1; ii<all_int.size(); ++ii)
        mean = mean + all_int[ii];
    mean.x = mean.x / (float)all_int.size();
    mean.y = mean.y / (float)all_int.size();
    return mean;
}

/* Function to recognize a digit in a binarized image using Tesseract 
 *  Note that we should limit Tesseract to look for digits only, but I didn't manage to do it from the C++ API... :)
 *  That's why we need to handle the 'I' as a '1', er similar...
 */
unsigned int recognize_digit(Mat& im,tesseract::TessBaseAPI& tess)
{
    tess.SetImage((uchar*)im.data, im.size().width, im.size().height, im.channels(), (int)im.step1());
    tess.Recognize(0);
    const char* out = tess.GetUTF8Text();
    if (out)
        if(out[0]=='1' or out[0]=='I' or out[0]=='i' or out[0]=='/' or out[0]=='|' or out[0]=='l' or out[0]=='t')
            return 1;
        else if(out[0]=='2')
            return 2;
        else if(out[0]=='3')
            return 3;
        else if(out[0]=='4')
            return 4;
        else if(out[0]=='5' or out[0]=='S' or out[0]=='s')
            return 5;
        else if(out[0]=='6')
            return 6;
        else if(out[0]=='7')
            return 7;
        else if(out[0]=='8')
            return 8;
        else if(out[0]=='9')
            return 9;
        else
            return 0;
    else
        return 0;
}




int main()
{
    /* Class to capture the webcam feed */
    VideoCapture capture(0);
    
    /* Create the window */
    string window_name = "Sudoku AR Solver";
    namedWindow(window_name, CV_WINDOW_KEEPRATIO);
    
    /* Frame containers */
    Mat raw_frame,frame,frame_gray,blurred_frame_gray,detected_edges,color_edges;
    
    /* Start Tesseract OCR, we will use it to recognize digits */
    tesseract::TessBaseAPI tess;
    if (tess.Init("/opt/local/share/tessdata/", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
    
    /* Global loop */
    while(true)
    {
        /* Capture one frame from the webcam */
        capture >> frame;
        if (frame.empty())
            break;
        
        /* Some constants */
        size_t sx = frame.cols;
        size_t sy = frame.rows;
        if (mode==0)
        {
            /* Show the result */
            imshow(window_name, frame);
        }
        else
        {
            /* To gray and blur for the Canny */
            cvtColor(frame, frame_gray, CV_BGR2GRAY);
            blur( frame_gray, blurred_frame_gray, Size(3,3) );
            
            /* Canny edge detector */
            Canny( blurred_frame_gray, detected_edges, lowThreshold, lowThreshold*ratio2, kernel_size );
            if (mode==1)
            {
                /* Show the result */
                imshow(window_name, detected_edges);
            }
            else
            {
                /* Detect lines by Hough */
                vector<Vec2f> det_lines;
                HoughLines(detected_edges, det_lines, 2, CV_PI/180, 300, 0, 0 );
                
                /* Extract segments out of the lines to paint them on the frame - OpenCV expects a segment */
                vector<Line> lines(det_lines.size());
                for( size_t ii = 0; ii < det_lines.size(); ++ii )
                {
                    float rho = det_lines[ii][0], theta = det_lines[ii][1];
                    double a = cos(theta), b = sin(theta);
                    double x0 = a*rho, y0 = b*rho;
                    lines[ii].e1.x = cvRound(x0 + 2000*(-b));
                    lines[ii].e1.y = cvRound(y0 + 2000*(a));
                    lines[ii].e2.x = cvRound(x0 - 2000*(-b));
                    lines[ii].e2.y = cvRound(y0 - 2000*(a));
                    lines[ii].id = ii;
                }

                /* Separate them into horizontal and vertical  by setting a threshold on the slope*/
                vector<pair<double,Line>> horiz;
                vector<pair<double,Line>> verti;
                vector<pair<double,Line>> other;
                for( size_t ii = 0; ii < lines.size(); ++ii )
                    if(det_lines[ii][1]<CV_PI/20 or det_lines[ii][1]>CV_PI-CV_PI/20) /* Vertical if close to 180 deg or to 0 deg */
                        verti.push_back(make_pair(det_lines[ii][0],lines[ii]));
                    else if(abs(det_lines[ii][1]-CV_PI/2)<CV_PI/20)                  /* Horizontal if close to 90 deg */
                        horiz.push_back(make_pair(det_lines[ii][0],lines[ii]));
                    else
                        other.push_back(make_pair(det_lines[ii][0],lines[ii]));
                
                /* Sort them in order of rho */
                std::sort(verti.begin(), verti.end(), [](const std::pair<double,Line> &left, const std::pair<double,Line> &right) {return left.first < right.first;});
                std::sort(horiz.begin(), horiz.end(), [](const std::pair<double,Line> &left, const std::pair<double,Line> &right) {return left.first < right.first;});
                
                /* And now store their relative position (order) in the frame sorted by rho */
                for(std::size_t ii=0; ii<verti.size(); ++ii)
                    verti[ii].second.hv_id = ii;
                for(std::size_t ii=0; ii<horiz.size(); ++ii)
                    horiz[ii].second.hv_id = ii;
                
                if (mode==2)
                {
                    /* Paint them on the frame */
                    for( auto it: verti)
                        line( frame, it.second.e1, it.second.e2, Scalar(  0,  0,255), 2, CV_AA);
                    for( auto it: horiz)
                        line( frame, it.second.e1, it.second.e2, Scalar(255,  0,  0), 2, CV_AA);
                    for( auto it: other)
                        line( frame, it.second.e1, it.second.e2, Scalar(  0,  0,  0), 1, CV_AA);
                }
                else
                {
                    /* Compute pairwise intersections between vertical and horizontal lines */
                    for(auto& vert_it: verti)
                    {
                        for(auto& hori_it: horiz)
                        {
                            Point2f inters;
                            if(intersection(vert_it.second.e1, vert_it.second.e2,
                                            hori_it.second.e1, hori_it.second.e2, inters))
                            {
                                if(inters.x>=0 and inters.x<sx and inters.y>=0 and inters.y<sy)
                                {
                                    vert_it.second.intersections.insert(make_pair(inters.y,hori_it.second.hv_id));
                                    hori_it.second.intersections.insert(make_pair(inters.x,vert_it.second.hv_id));
                                }
                            }
                        }
                    }
                
                    /* Scan one line in the center (less likely to be erroneous) and classify the orthogonal lines */
                    vector<set<size_t>> sel_v;
                    bool good1 = classify_lines(horiz,verti,sel_v);
                    
                    vector<set<size_t>> sel_h;
                    bool good2 = classify_lines(verti,horiz,sel_h);
                    
                    if (good1 and good2)
                    {
                        /* Find the corner points of the cells */
                        vector<vector<Point2f>> corners(10,vector<Point2f>(10));
                        for(std::size_t ii=0; ii<10; ++ii)
                            for(std::size_t jj=0; jj<10; ++jj)
                                corners[ii][jj] = mean_intersection(sel_h[ii],sel_v[jj],horiz,verti);
                        
                        if (mode==3)
                        {
                            /* Plot the corners */
                            for(std::size_t ii=0; ii<10; ++ii)
                                for(std::size_t jj=0; jj<10; ++jj)
                                    paint_point(frame, corners[ii][jj]);
                        }
                        else
                        {
                            /* Create the boxes of the cells */
                            float reduce_percent = 0.6;
                            vector<vector<pair<Point2f,Point2f>>> boxes(9,vector<pair<Point2f,Point2f>>(9));
                            for(std::size_t ii=0; ii<9; ++ii)
                                for(std::size_t jj=0; jj<9; ++jj)
                                {
                                    Point2f ul = corners[ii][jj];
                                    Point2f dr = corners[ii+1][jj+1];
                                    
                                    /* We reduce the size a certain percentage to avoid borders */
                                    float w = (dr.x - ul.x)*reduce_percent;
                                    float h = (dr.y - ul.y)*reduce_percent;
                                    float c_x = (dr.x + ul.x)/2;
                                    float c_y = (dr.y + ul.y)/2;
                                    ul.x = c_x-w/2;
                                    ul.y = c_y-h/2;
                                    dr.x = c_x+w/2;
                                    dr.y = c_y+h/2;
                                    
                                    boxes[ii][jj].first = ul;
                                    boxes[ii][jj].second = dr;
                                }
                            
                            if (mode==4)
                            {
                                /* Plot the boxes */
                                for(std::size_t ii=0; ii<9; ++ii)
                                    for(std::size_t jj=0; jj<9; ++jj)
                                        rectangle(frame, boxes[ii][jj].first, boxes[ii][jj].second, Scalar(0,0,255) );
                            }
                            else
                            {
                                /* Get the image of the Sudoku full box by getting the first and last grids
                                 *  - ulx: Up Left X
                                 *  - uly: Up Left Y
                                 *  - drx: Down Right X
                                 *  - dry: Down Right Y
                                 */
                                unsigned int ulx = round(min(corners[0][0].x,corners[9][0].x));
                                unsigned int uly = round(min(corners[0][0].y,corners[0][9].y));
                                
                                unsigned int drx = round(max(corners[0][9].x,corners[9][9].x));
                                unsigned int dry = round(max(corners[9][0].y,corners[9][9].y));
                                
                                /* This is to be robust against some degenerate cases */
                                if(ulx>sx or uly>sy or drx>sx or dry>sy)
                                    continue;
                                
                                /* Crop the image */
                                Mat sudoku_box(frame_gray, cv::Rect(ulx, uly,
                                                                    drx-ulx,
                                                                    dry-uly));
                                
                                /* Apply local thresholding */
                                Mat sudoku_th = sudoku_box.clone();
                                adaptiveThreshold(sudoku_box, sudoku_th, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);
                                
                                /* To adjust parameters, we can write the image */
                                // imwrite( "SudokuTh.png", sudoku_th );
                                
                                /* Process all boxes and classify whether they are empty (we'll say 0) or there is a number 1-9 */
                                vector<vector<unsigned int>> rec_digits(9,vector<unsigned int>(9));
                                for(std::size_t ii=0; ii<9; ++ii)
                                {
                                    for(std::size_t jj=0; jj<9; ++jj)
                                    {
                                        /* Get the square as an image */
                                        Mat digit_box(sudoku_th, cv::Rect(round(boxes[ii][jj].first.x)-ulx, round(boxes[ii][jj].first.y)-uly,
                                                                          round(boxes[ii][jj].second.x-boxes[ii][jj].first.x),
                                                                          round(boxes[ii][jj].second.y-boxes[ii][jj].first.y)));
                                        
                                        /* Recognize the digit using the OCR */
                                        rec_digits[ii][jj] = recognize_digit(digit_box, tess);
                                        
                                        /* To debug, we can write the binarized small images */
                                        // stringstream ss;
                                        // ss << "/Users/jpont/Downloads/Sudoku_" << ii+1 << "_" << jj+1 << ".png";
                                        // imwrite(ss.str().c_str(), digit_box );
                                    }
                                }
                                
                                if (mode==5)
                                {
                                    /* Plot the recognized numbers on top of the image */
                                    for(std::size_t ii=0; ii<9; ++ii)
                                        for(std::size_t jj=0; jj<9; ++jj)
                                            if (rec_digits[ii][jj]!=0)
                                            {
                                                Point text_pos(boxes[ii][jj].first.x+(boxes[ii][jj].second.x-boxes[ii][jj].first.x)/5,
                                                               boxes[ii][jj].second.y-(boxes[ii][jj].second.y-boxes[ii][jj].first.y)/5);
                                                stringstream ss;
                                                ss << (int)rec_digits[ii][jj];
                                                putText(frame, ss.str(), text_pos, CV_FONT_HERSHEY_DUPLEX, /*Size*/1,
                                                        Scalar(0,0,255), /*Thickness*/ 1, 8);
                                            }
                                }
                                else
                                {
                                    /* Create the Sudoku class */
                                    const int N = 3;
                                    Sudoku<N> sudoku;
                                    
                                    /* Set the recognized digits */
                                    for(std::size_t ii=0; ii<N*N; ++ii)
                                        for(std::size_t jj=0; jj<N*N; ++jj)
                                            sudoku.set_value(ii, jj, rec_digits[ii][jj]);
                                    
                                    /* Let's try to solve it. If we solved it, plot the 
                                     * numbers in the gaps using augmenting reality */
                                    if(sudoku.solve())
                                        for(std::size_t ii=0; ii<N*N; ++ii)
                                            for(std::size_t jj=0; jj<N*N; ++jj)
                                                if (rec_digits[ii][jj]==0)
                                                {
                                                    Point text_pos(boxes[ii][jj].first.x +(boxes[ii][jj].second.x-boxes[ii][jj].first.x)/5,
                                                                   boxes[ii][jj].second.y-(boxes[ii][jj].second.y-boxes[ii][jj].first.y)/5);
                                                    stringstream ss;
                                                    ss << (int)sudoku.get_value(ii,jj);
                                                    putText(frame, ss.str(), text_pos, CV_FONT_HERSHEY_DUPLEX, /*Size*/1,
                                                            Scalar(0,0,255), /*Thickness*/ 1, 8);
                                                }
                            
                                }
                            }
                        }
                    }
                }
                
                /* Show the result */
                imshow(window_name, frame);
            }
        }
        
        
        /* Wait for a key */
        char key = (char)waitKey(5);
        switch (key)
        {
            case '0':
                mode = 0;
                break;
            case '1':
                mode = 1;
                break;
            case '2':
                mode = 2;
                break;
            case '3':
                mode = 3;
                break;
            case '4':
                mode = 4;
                break;
            case '5':
                mode = 5;
                break;
            case '6':
                mode = 6;
                break;
            case 27: //escape key
                return 0;
            default:
                break;
        }
    }
    return 0;
}


