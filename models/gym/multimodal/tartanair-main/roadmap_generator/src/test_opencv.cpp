#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

# include <iostream>

using namespace std;
using namespace cv;
void printMat(Mat m, int xlen, int ylen)
{
    for(int i=0; i<xlen; i++)
    {
        for(int j=0; j<ylen; j++)
        {
            cout<<m.at<float>(i,j)<<" ";
        }
        cout<<endl;
    }
}

int main(int argc, char** argv){
    float data[] = {0.1 ,0.2, 0.3,
                    0.4, 0.5, 0.6,
                    0.7, 0.8, 0.9};
    Mat m(3, 3, CV_32FC1, data);
    printMat(m*2, 3, 3);
    m = m * 2;
    threshold( m, m, 0.55, 0.55, 2 );
    printMat(m, 3, 3);
    return(0);
}


