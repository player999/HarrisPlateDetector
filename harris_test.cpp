#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <boost/filesystem.hpp>
#include <string>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

/// Global variables
Mat src, binary;

#define GRAN_LEVELS (1024)

#define P6

#ifdef P1
float thresh = 0.28f;
int max_thresh = 255;
int median_kernel = 3;
int apertureSize = 25;
int blockSize = 6;
double k = 0.02;
double sigma = 0.8;
float sharko = 0.8f;
#endif

#ifdef P2
float thresh = 0.42f;
int max_thresh = 255;
int median_kernel = 5;
int apertureSize = 11;
int blockSize = 4;
int gkernel = 21;
double k = 0.02;
double sigma = 0.8;
float sharko = 0.8;
#endif

#ifdef P3
float thresh = 0.25f;
int max_thresh = 255;
int median_kernel = 5;
int apertureSize = 11;
int blockSize = 8;
int gkernel = 21;
double k = 0.02;
double sigma = 0.8;
float sharko = 0.8f;
#endif

#ifdef P4
float thresh = 0.89f;
int max_thresh = 255;
int median_kernel = 5;
int apertureSize = 11;
int blockSize = 2;
int gkernel = 21;
double k = 0.02;
double sigma = 0.8;
float sharko = 0.8f;
#endif

#ifdef P5
float thresh = 0.44f;
int max_thresh = 255;
int median_kernel = 3;
int apertureSize = 11;
int blockSize = 6;
int gkernel = 21;
double k = 0.04;
double sigma = 0.8;
float sharko = 0.8f;
#endif

#ifdef P6
float thresh = 0.41f;
int max_thresh = 255;
int median_kernel = 7;
int apertureSize = 1;
int blockSize = 6;
int gkernel = 5;
double k = 0.0358;
double sigma = 0.1;
float sharko = 0.4312f;
#endif

static char wname[] = "Harris";

static double thresh_min = 0.0;
static double thresh_max = 1.0;
static int mkernel_min = 1;
static int mkernel_max = 21;
static int aperture_min = 3;
static int aperture_max = 29;
static int block_min = 2;
static int block_max = 20;
static int gkernel_min = 3;
static int gkernel_max = 27;
static double kmin = 0.0;
static double kmax = 0.1;
static double sigma_min = 0.1;
static double sigma_max = 0.9;
static double sharko_min = 0.0;
static double sharko_max = 0.5;

static int tthresh;
static int tmkernel;
static int taperture;
static int tblock;
static int tgkernel;
static int tk;
static int tsigma;
static int tsharko;

#define GRAN 500

vector<Point> calculatePoints(Mat &input);
Mat drawPoints(Mat input, vector<Point> &points);
Mat drawLines(Mat input, vector< tuple<Point, Point> > lin);
Mat drawRects(Mat input, vector< tuple< tuple<Point, Point>, tuple<Point, Point> > > rects);
void cornerHarris_demo( int, void* );
void calculateHarris(Mat &input, Mat &output);
vector<tuple<Point,Point>> find_horizontal_lines(vector<Point> &points,
  double delta, double Dmin, double Dmax);
vector< tuple< tuple<Point, Point>, tuple<Point, Point> > >
find_parallel_horizontal_lines(vector< tuple<Point,Point> > hlines,
  double len_delta, double loc_delta, double Dmin, double Dmax);

static inline void process_changes()
{
  calculateHarris(src, binary);
  vector<Point> centers = calculatePoints(binary);
  Mat circles = drawPoints(src, centers);
  auto hlines = find_horizontal_lines(centers, 3, 80, 150);
  //circles = drawLines(circles, hlines);
  auto plines = find_parallel_horizontal_lines(hlines, 5, 5, 15, 30);
  circles = drawRects(circles, plines);
  Mat matArray[] = {binary, circles};
  Mat out;
  hconcat(matArray, 2, out);
  imshow(wname, out);

}

void calculateHarris(Mat &input, Mat &output);

void on_thresh( int, void* )
{
 thresh = thresh_min + ((double) tthresh / (double)GRAN) * (thresh_max - thresh_min);
 process_changes();
}

void on_median( int, void* )
{
 median_kernel = (tmkernel % 2)?tmkernel:tmkernel + 1;
 process_changes();
}

void on_aperture( int, void* )
{
 apertureSize = (taperture % 2)?taperture:taperture + 1;
 process_changes();
}

void on_block( int, void* )
{
 blockSize = tblock;
 process_changes();
}

void on_gkernel( int, void* )
{
 gkernel = (tgkernel % 2)?tgkernel:tgkernel + 1;
 process_changes();
}

void on_k( int, void* )
{
 k = kmin + ((double) tk / (double)GRAN) * (kmax - kmin);
 process_changes();
}

void on_sigma( int, void* )
{
 sigma = sigma_min + ((double) tsigma / (double)GRAN) * (sigma_max - sigma_min);
 process_changes();
}

void on_sharko( int, void* )
{
 sharko = sharko_min + ((double) tsharko / (double)GRAN) * (sharko_max - sharko_min);
 process_changes();
}

// void nextImage( int, void* )
// {
//   if (diterator != boost::filesystem::directory_iterator{}) {
//     const boost::filesystem::directory_entry& entry = *diterator;
//     src = imread(entry.path().string(), 1 );
//     cvtColor( src, src, CV_BGR2GRAY );
//     process_changes();
//     diterator++;
//   }
// }

// void prevImage( int, void* )
// {
//   if (diterator != boost::filesystem::directory_iterator{}) {
//     const boost::filesystem::directory_entry& entry = *diterator;
//     src = imread(entry.path().string(), 1 );
//     cvtColor( src, src, CV_BGR2GRAY );
//     process_changes();
//     diterator--;
//   }
// }


void calculateHarris(Mat &input, Mat &output)
{
    Mat gausskernel, median, floating, gaussian, unsharp_mask, sharp;
    //Preprocess
    if (median_kernel >= 3) {
      medianBlur(input, median, median_kernel);
    }
    gausskernel = getGaussianKernel(gkernel, sigma, CV_32F);
    median.convertTo(floating, CV_32FC1);
    floating /= 255;
    filter2D(floating, gaussian, -1, gausskernel, Point(-1,-1), 0, BORDER_DEFAULT);
    unsharp_mask = floating - gaussian;
    sharp = floating + sharko * unsharp_mask;
    normalize(sharp, sharp, 0, 1, NORM_MINMAX, CV_32FC1, Mat());

    //Do HARRIS
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris(sharp, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    normalize(dst, output, 0, 1, NORM_MINMAX, CV_32FC1, Mat() );
    //void calcHist( const int* histSize, const float** ranges,
    MatND hist;
    int channels[] = {0};
    int nlevels = GRAN_LEVELS;
    float range[] = {0,1};
    const float* ranges[] = {range};
    calcHist(&output, 1, channels, Mat(), hist, 1, &nlevels, ranges, true, false);
    Point max_idx;
    minMaxLoc(hist, NULL, NULL, NULL, &max_idx);
    thresh = 1.0f / GRAN_LEVELS * (max_idx.y + 2);
    output = output > thresh;
}

vector<Point> calculatePoints(Mat &input)
{
  vector< vector<Point> >contours;
  findContours(input.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  vector<Point> centers;
  for (int i = 0; i < contours.size(); i++)
  {
    int max_x = 0, max_y = 0;
    int min_x = input.rows - 1, min_y = input.cols - 1;
    for (int j = 0; j < contours[i].size(); j++) {
      if (contours[i][j].x > max_x) max_x = contours[i][j].x;
      if (contours[i][j].x < min_x) min_x = contours[i][j].x;
      if (contours[i][j].y > max_y) max_y = contours[i][j].y;
      if (contours[i][j].y < min_y) min_y = contours[i][j].y;
    }
    int center_x, center_y;
    center_x = (max_x + min_x) / 2;
    center_y = (max_y + min_y) / 2;
    centers.push_back(Point(center_x, center_y));
  }
  return centers;
}

Mat drawPoints(Mat input, vector<Point> &points)
{
  Mat retimage = input.clone();
  for (int i; i < points.size(); i++) {
    circle(retimage, points[i], 5, Scalar(0, 0, 0, 0), 2);
  }
  return retimage;
}

Mat drawLines(Mat input, vector< tuple<Point, Point> > lin) {
  for (int i = 0; i < lin.size(); i++) {
    line(input, get<0>(lin[i]), get<1>(lin[i]), (0,0,0), 1);
  }
  return input;
}

Mat drawRects(Mat input, vector< tuple< tuple<Point, Point>, tuple<Point, Point> > > rects)
{
  for (int i = 0; i < rects.size(); i++) {
    tuple< tuple<Point, Point>, tuple<Point, Point> > r = rects[i];
    Point p1, p2;
    p1.x = min(min(get<0>(get<0>(r)).x, get<0>(get<1>(r)).x),
               min(get<1>(get<0>(r)).x, get<1>(get<1>(r)).x));
    p2.x = max(max(get<0>(get<0>(r)).x, get<0>(get<1>(r)).x),
               max(get<1>(get<0>(r)).x, get<1>(get<1>(r)).x));
    p1.y = min(min(get<0>(get<0>(r)).y, get<0>(get<1>(r)).y),
               min(get<1>(get<0>(r)).y, get<1>(get<1>(r)).y));
    p2.y = max(max(get<0>(get<0>(r)).y, get<0>(get<1>(r)).y),
               max(get<1>(get<0>(r)).y, get<1>(get<1>(r)).y));
    rectangle(input, p1, p2, Scalar(0, 0, 0, 0));
  }
  return input;
}

vector< tuple<Point,Point> > find_horizontal_lines(vector<Point> &points,
  double delta, double Dmin, double Dmax)
{
  auto distance = [](Point const &p1, Point const &p2)
    { return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)); };

  vector< tuple<Point, Point> > hlines;
  for (int i = 0; i < points.size(); i++) {
    vector<Point> goodpoints;
    copy_if(points.begin(), points.end(), back_inserter(goodpoints),
      [points, i, delta](Point const &p){return (abs(points[i].y - p.y) < delta);});
    auto nd = remove_if(goodpoints.begin(), goodpoints.end(),
      [Dmin, Dmax, points, i, distance](Point const &p)
        {double d = distance(points[i], p); return ((d < Dmin) || (d > Dmax));});
    goodpoints.erase(nd, goodpoints.end());
    vector< tuple<Point, Point> > pieces;
    pieces.resize(goodpoints.size());
    transform(goodpoints.begin(), goodpoints.end(), pieces.begin(),
      [points, i](Point const &p) {
          tuple<Point, Point> t;
          if (points[i].x > p.x) t = tuple<Point,Point>(p, points[i]);
          else t = tuple<Point,Point>(points[i], p);
          return t;
      }
    );
    hlines.insert(hlines.end(), pieces.begin(), pieces.end());
  }
  return hlines;
}

vector< tuple< tuple<Point, Point>, tuple<Point, Point> > >
find_parallel_horizontal_lines(vector< tuple<Point,Point> > hlines,
  double len_delta, double loc_delta, double Dmin, double Dmax)
{
  vector< tuple< tuple<Point, Point>, tuple<Point, Point> > > par;
  vector<double> lens;
  lens.resize(hlines.size());
  transform(hlines.begin(), hlines.end(), lens.begin(),
    [](tuple<Point,Point> const &l) {return abs(get<0>(l).x - get<1>(l).x);});

  vector<double> locs;
  locs.resize(hlines.size());
  transform(hlines.begin(), hlines.end(), locs.begin(),
    [](tuple<Point,Point> const &l) {return (get<0>(l).x + get<1>(l).x) / 2;});

  auto distance = [](tuple<Point,Point> const &l1, tuple<Point,Point> const &l2) {
    return abs((get<0>(l1).y + get<1>(l1).y) / 2 - (get<0>(l2).y + get<1>(l2).y) / 2);
  };

  for (int i = 0; i < hlines.size(); i++) {
    for (int j = i + 1; j < hlines.size(); j++) {
      bool ok_len = fabs(lens[i] - lens[j]);
      bool ok_locs = fabs(locs[i] - locs[j]);
      double d = distance(hlines[i], hlines[j]);
      bool ok_dist = (d > Dmin) && (d < Dmax);
      bool valid = ok_len && ok_locs && ok_dist;
      if (valid) par.push_back(tuple< tuple<Point, Point>, tuple<Point, Point> >(hlines[i], hlines[j]));
    }
  }
  return par;
}

/** @function main */
int main( int argc, char** argv )
{

  namedWindow(wname, 1);

  tthresh = (thresh - thresh_min) * GRAN / (thresh_max - thresh_min);
  createTrackbar("Threshold", wname, &tthresh, GRAN, on_thresh);
  tmkernel = median_kernel - mkernel_min;
  createTrackbar("Median", wname, &tmkernel, mkernel_max - mkernel_min, on_median);
  taperture = apertureSize - aperture_min;
  createTrackbar("Aperture Size", wname, &taperture, aperture_max - aperture_min, on_aperture);
  tblock = blockSize - block_min;
  createTrackbar("Block Size", wname, &tblock, block_max - block_min, on_block);
  tgkernel = gkernel - gkernel_min;
  createTrackbar("Gaussian kernel", wname, &tgkernel, gkernel_max - gkernel_min, on_gkernel);
  tk = (k - kmin) * GRAN / (kmax - kmin);
  createTrackbar("K", wname, &tk, GRAN, on_k);
  tsigma = (sigma - sigma_min) * GRAN / (sigma_max - sigma_min);
  createTrackbar("Sigma", wname, &tsigma, GRAN, on_sigma);
  tsharko = (sharko - sharko_min) * GRAN / (sharko_max - sharko_min);
  createTrackbar("Sharko", wname, &tsharko, GRAN, on_sharko);

  path image_path(argv[1]);
  directory_iterator diterator{image_path};
  for (directory_iterator it = directory_iterator(diterator); it != directory_iterator(); it++ ) {
    src = imread((*it).path().string(), 1 );
    cvtColor( src, src, CV_BGR2GRAY );

    cornerHarris_demo( 0, 0 );

    int keycode = waitKey(0);
    cout << *it << endl;
    if (27 == keycode) break;
  }

  cout << "Threshold " << thresh << endl;
  cout << "Median " << median_kernel << endl;
  cout << "Aperture Size " << apertureSize << endl;
  cout << "Block Size " << blockSize << endl;
  cout << "Gaussian kernel " << gkernel << endl;
  cout << "K " << k << endl;
  cout << "Sigma " << sigma << endl;
  cout << "Sharko " << sharko << endl;

  return(0);
}

/** @function cornerHarris_demo */
void cornerHarris_demo( int, void* )
{
  process_changes();
}