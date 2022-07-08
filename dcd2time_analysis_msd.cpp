/**
 * TITLE:
 * dcd2time_analysis.cpp
 *
 * AUTHOR:
 * E. Aaron
 *
 * MODIFIED:
 * 16 June 2022
 *
 * DESCRIPTION:
 * Reads a single thermal history recorded as a series of 1 or more DCD trajectory files,
 * and calculates the following (time-averaged) 2-time correlations:
 * 1. Mean Square Displacement
 * 2. Self-Intermediate Scattering Function (f_s) in 3 dimensions
 * 3. Overlap Function (f_o)
 * 4. Static structure factor (s_4) as a function of various wave-vector values and particle positions
 *
 * The code takes the following as input:
 * 1. number of files (int)
 * 2. first initial time to use for averaging (int)
 * 3. shortest time interval to analyze (int)
 * 4. longest time interval to analyze (int)
 * 5. total number of time intervals (int)
 * 6. choice of whether or not to specify time interval between initial times (y/n)
 *  [if no interval is specified, the program automatically uses non-overlapping time intervals]
 * 6.5 if (6) is y, the time interval between initial times to be used (int)
 * 7. 'q' parameter [wave vector] of self-intermediate scattering function (double)
 * 8. 'a' parameter [packing fraction] of overlap function (double)
 * 9. maximum wave vector magnitude to use in argument of s_4
 * 10. full absolute path to directory containing trajectory files to be analyzed (string)
 *  [ex: /project/data/05/run1]
 * in that order. These can be passed to the compiled program by storing them in an input file,
 * separated by newlines, and passing that file as an argument to the program at execution.
 *
 * NOTE:
 * This program is expanded & heavily revised from Elijah Flenner, et al. "template.c".
 * Flenner's DCD reading header file, "dcd.h", MUST be #included for the program to
 * compile & run.
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstring>
#include <ctime>
#include <complex>
#include "dcd.h"
//#include "sort.h"
//#include "quicksort.h"

using namespace std;

long int read_dcd_head(FILE *input, int *N,int flag); // Read in file info
int gdcdp(float *x, float *y, float *z, char file[],long int n,int flag,long int *pos, int wcell); // Read in data
long int dcd_info(char file[], int *N,int *nset, int *tbsave, float *tstep, int *wcell); // Read in parameters
// CM coordinate calculator function
void sortqs(vector<double> &qx, vector<double> &qy, vector<double> &qz, vector<int> itable);
float cm(float *x, int N);
void printParams(string path, int nfiles, int tfirst, int tint_sm, int tint_lg, int dtsave, int ntints, char t0_choice, int t0_dif, float q, float a, double qmax, double dt);

const int SIZE = 500;
const complex<double> I = sqrt(-1);

int main() {

    time_t t1 = time(0);
    cout << "Initializing..." << endl;

    int nfiles = 1, t0_sm = 0, tint_sm = 1, tint_lg = 1, ntints = 2, nt0s_max = 1;
    float q = 1, a = 1, qmax = 1, rho = 1.2; // params of fs, fo
    float tstep, dcdtimes, delta_t; // tstep only used to fill args of dcd_info, dcdtimes is last time recorded in DCD files
    int cellsize, filesnapshots; // cell size for data reading, # of particle snapshots per file
    int N, dtsave, t0_dif = 1; // total # of particles, time between recorded steps, initial time difference
    char t0_choice_char;
    bool t0_choice = false;
    string path;

// ========================================================================== //
// User Input
// ========================================================================== //

    cout << "Enter one at a time:" << endl;
    cout << "# of files | first initial time | smallest time interval | longest time interval | total # of time intervals" << endl;
    cin >> nfiles >> t0_sm >> tint_sm >> tint_lg >> ntints;
    cout << nfiles << " " << t0_sm << " " << tint_sm << " " << tint_lg << " " << ntints << endl;
    cout << "Specify difference between initial times? (y / n) " << endl;
    cin >> t0_choice_char;
    if(t0_choice_char == 'y' || t0_choice_char == 'Y') {
        t0_choice = true;
        cout << "Enter time difference: ";
        cin >> t0_dif;
    }
    cout << "Enter one at a time:" << endl;
    cout << "'q' parameter | 'a' parameter | max q (s4) | dt" << endl;
    cin >> q >> a >> qmax >> delta_t;
    double a2 = pow(a,2);
    cout << "Enter path to data directory (without ending /): " << endl;
    cin >> path;

// ========================================================================== //
// Preparatory
// ========================================================================== //
    cout << "Defining logistical variables..." << endl;
    cout << "tint_lg = " << tint_lg << endl;
    string filenames_str[SIZE];
    char filename[SIZE];
    for(int i = 0; i < nfiles; i++) {
        filenames_str[i] = path + "/traj" + to_string(i+1) + ".dcd";
    }

// ========================================================================== //
//      Read params from dcd files

    int currsnapshots; //# of snapshots in file currently being read
    dcdtimes = 0;
    for(int i = 0; i < nfiles; i++) {
        strcpy(filename, filenames_str[i].c_str());
        dcd_info(filename, &N, &currsnapshots, &dtsave, &tstep, &cellsize);
        if(i == 0) filesnapshots = currsnapshots;
        if(currsnapshots != filesnapshots) {
            cout << "Files have inconsistent numbers of snapshots." << endl;
            exit(1);
        }
        dcdtimes += currsnapshots;
    }
    memset(&filename, 0, sizeof(filesnapshots));

// ========================================================================== //
//      Recast in units of dtsave (time between snapshots)
    if(t0_choice == false) t0_dif = dtsave*delta_t;
    else if(t0_dif < dtsave*delta_t){
        cout << "ERROR: initial time difference " << t0_dif << " too small. Must be at least " << dtsave*delta_t << endl;
        exit(0);
    }
    int t0_snapshot_sm = t0_sm / dtsave / delta_t;
    int dt_snapshot_sm = tint_sm / dtsave / delta_t;
    int dt_snapshot_lg = tint_lg / dtsave / delta_t;
    cout << "dt_snapshot_lg = " << dt_snapshot_lg << endl;
    if(dt_snapshot_lg > dcdtimes) dt_snapshot_lg = dcdtimes;
    int t0_snapshot_dif = t0_dif / dtsave / delta_t;
    //remainder of code is written in terms of snapshots

// ========================================================================== //
//      Calculate snapshot differences

    cout << "Calculating time intervals..." << endl;
    //define params & set initial values
    vector<int> dt_snapshots; //vect of time snapshot (snapshot) differences corresponding to user specified time differences
    double ratio; //geometric series base for logarithmic spacing of snapshot differences
    //int t;
    int snapshot = dt_snapshot_sm; //starting snapshot
    int ratiosm = dt_snapshot_sm; //starting ratio
    if (dt_snapshot_sm == 0) ratiosm = 1; //keep ratio denominator nonzero
    //fill dt_snapshots
    cout << "t0_dif = " << t0_dif << endl;
    cout << "dcdtimes = " << dcdtimes << endl;
    cout << "ratiosm = " << ratiosm << endl;
    cout << "(1.0 * dt_snapshot_lg) = " << (1.0 * dt_snapshot_lg) << endl;
    cout << "(1.0 * ratiosm) = " << (1.0 * ratiosm) << endl;
    cout << "1.0 / ntints = " << 1.0 / ntints << endl;
    cout << "pow((1.0 * dt_snapshot_lg) / (1.0 * ratiosm), 1.0 / ntints) = " << pow((1.0 * dt_snapshot_lg) / (1.0 * ratiosm), 1.0 / ntints) << endl;

    dt_snapshots.push_back(snapshot);
    ratio = pow((1.0 * dt_snapshot_lg) / (1.0 * ratiosm), 1.0 / ntints);
    cout << "Log spacing ratio: " << ratio << endl;
    int ti = 1;
      //int tsaved = 1;
    cout << "Saving time differences..." << endl;
    while(snapshot < dt_snapshot_lg) {
        snapshot = ratiosm * pow(ratio,ti);
        if(snapshot > dt_snapshot_lg) break;
        else if(snapshot > dt_snapshots.back()) {
            dt_snapshots.push_back(snapshot);
        }
        ti++;
    }
    ntints = dt_snapshots.size();
    nt0s_max = (dcdtimes*nfiles - t0_snapshot_sm) / t0_snapshot_dif;
    //int nt0s_max2 = nt0s_max*nt0s_max;
    cout << "t0_snapshot_dif = " << t0_snapshot_dif << endl;
    cout << "t0_snapshot_sm = " << t0_snapshot_sm << endl;
    cout << "dtsave = " << dtsave << endl;
    cout << "dcdtimes = " << dcdtimes << endl;
    cout << "nt0s_max = " << nt0s_max << endl;

// ========================================================================== //
//      Calculate S4 params
    //cout << "phi=" << phi << endl;
    float L = pow(N/rho,0.333333333); //box length
    cout << "L(" << N << ") = " << L << endl;
/*
    //cout << "L(" << phi << ") = " << L << endl;
    cout << "Calculating s4 parameters..." << endl;
    //store all allowed combinations of qx, qy, qz such that 0 < |q| < 1
    float L = pow(0.980176*N/phi,0.333333333); //box length
    cout << "L=" << L << endl;
    double pi_2 = 6.28318530718;
    cout << "2pi/L=" << pi_2/L << endl;
    double qsquare = 0; //q dot q
    //vector< vector<double> > q_vect; //1d vector of 3d q vectors
    vector<double> qmags; //vector of squared magnitudes of q
    //vector<double> q1(3); //a single 3d q vector
    vector<double> qx_vect,qy_vect,qz_vect; //vectors of components
    double qx,qy,qz; //components
    double qmax2 = qmax*qmax;
    cout << "qmax^2=" << qmax2 << endl;
    double nmax = qmax / pi_2 * L; //magnitude of largest & smallest indices
    cout << "Calculating qs..." << endl;
    //create vectors of qs (x,y,z,magnitude^2)
    for(int i = 0; i <= nmax; i++) {
        qx = i*pi_2/L; //choose an x value
        for(int j = 0; j <= i; j++) {
            qy = j*pi_2/L; //choose a y value
            for(int k = 0; k <= j; k++) {
                qz = k*pi_2/L; //choose a z value
                qsquare = qx*qx + qy*qy + qz*qz;
                if(qsquare < qmax2) {
                    cout << "q = {" << qx << " , " << qy << " , " << qz << "}";
                    qx_vect.push_back(qx);
                    qy_vect.push_back(qy);
                    qz_vect.push_back(qz);
                    qmags.push_back(qsquare);
                    cout << "qmag=" << qmags.back() << endl;
                }
            }
        }
    }
    int nqs = qmags.size();
    cout << "nqs = " << nqs << endl;
    cout << "Sorting qs..." << endl;
    //prepare to sort qs by magnitude
    vector<int> indices(nqs);
    for(int i = 0; i < nqs; i++) indices.at(i) = i;
    //quicksort magnitudes and indices to create index table (quicksort.h)
    quickSort(qmags,indices,0,nqs-1);
    printArray(qmags,indices,nqs);
    //sort q components based on index table (quicksort.h)
    sortqs(qx_vect, qy_vect, qz_vect, indices);
    //cout << "Sorted magnitudes: " << endl;
    /*
    for(int i = 0; i < nqs; i++) {
        cout << qx_vect.at(i)*qx_vect.at(i) + qy_vect.at(i)*qy_vect.at(i) + qz_vect.at(i)*qz_vect.at(i) << endl;
    }
    */

// ========================================================================== //
//      Initialize arrays

    cout << "Initializing arrays..." << endl;
    //positions
    float *x0, *y0, *z0, *xf, *yf, *zf;
    int size_xyz = N*sizeof(float);
    x0 = (float *) malloc(size_xyz);
	y0 = (float *) malloc(size_xyz);
	z0 = (float *) malloc(size_xyz);
	xf = (float *) malloc(size_xyz);
	yf = (float *) malloc(size_xyz);
	zf = (float *) malloc(size_xyz);

    //displacements
    float dx, dy, dz;

    //cm
    int size_cm = dcdtimes + 1; //for consistency with functions, should make size nt0s_max and index by t0_index rather than t0_snapshot.
    vector<float> cm_x(size_cm);
    vector<float> cm_y(size_cm);
    vector<float> cm_z(size_cm);

    //whether cm has been calculated yet for a given dcd snapshot (T/F)
    vector<bool> cm_times(size_cm);
    fill(cm_times.begin(),cm_times.end(),false);

    //self-intermediate scattering function
    int size_funcs = ntints + 1;
    vector<double> fs_x(size_funcs);
    vector<double> fs_y(size_funcs);
    vector<double> fs_z(size_funcs);

    //overlap function
    vector<double> fo(size_funcs);

    /*
    //s4 - index 1 is q, index 2 is time difference/interval
    //w term and t0 average
    vector< vector<complex<double> > > w(nqs);
    vector< complex<double> > w1(nt0s_max);
    complex<double> w11 = {0,0};
    fill(w1.begin(),w1.end(),w11);
    fill(w.begin(),w.end(),w1);
    vector< complex<double> > w_avg(nqs);
    fill(w_avg.begin(),w_avg.end(),w11);
    //w^2 term
    /*
    vector< vector<double> > w2(nqs);
    vector<double> w21(nt0s_max);
    fill(w21.begin(),w21.end(),0.0);
    fill(w2.begin(),w2.end(),w21);

    vector<double> w2_avg(nqs);
    fill(w2_avg.begin(),w2_avg.end(),0.0);
    //s4
    vector< vector<double> > s4(nqs);
    vector<double> s41(ntints);
    fill(s41.begin(),s41.end(),0.0);
    fill(s4.begin(),s4.end(),s41);
     */
    //mean square displacement
    vector<double> msd(size_funcs);

    //normalization counter for each initial time
    vector <int> normalization(size_funcs);



    //set all functions' initial values to zero
    fill(msd.begin(),msd.end(),0);
    fill(fs_x.begin(),fs_x.end(),0);
    fill(fs_y.begin(),fs_y.end(),0);
    fill(fs_z.begin(),fs_z.end(),0);
    fill(fo.begin(),fo.end(),0);

// ========================================================================== //
//      Misc

    //save analysis params to file
    printParams(path, nfiles, t0_snapshot_sm*dtsave*delta_t, dt_snapshot_sm*dtsave*delta_t, dt_snapshot_lg*dtsave*delta_t, dtsave*delta_t, ntints, t0_choice, t0_snapshot_dif*dtsave*delta_t, q, a, qmax, delta_t);
    //set up logistical parameters for main loops
    cout << "Reading data & evaluating functions..." << endl;
    int flag = 0; //dcd parameters found in Flenner's code (not used here)
    long int pos = -1;
    int whichfile; //file where current snapshot is located


// ========================================================================== //
// Main Calculations
// ========================================================================== //

    //Read data from dcd files, calculate function values from data, & store in arrays
    vector<int> nt0s(ntints);
    //Loop over snapshot differences
    for(int cindex = 0; cindex < ntints; cindex++) {
        nt0s.at(cindex) = 0;
        cout << "Time interval " << cindex + 1 << " of " << ntints << " : " << dt_snapshots.at(cindex)*dtsave*delta_t << endl;
      //s4
      //double s41 = 0;
    // ========================================================================== //
    //      averaging
      //non-overlapping vs overlapping averaging:
      //if fixed t0 difference (overlapping) was not specified by user, use current snapshot difference as *initial snapshot* spacing also (non-overlapping)
        if(t0_choice == false) {
            if(cindex == 0) t0_snapshot_dif = 1;
            else t0_snapshot_dif = dt_snapshots.at(cindex);
        }
        int t0_snapshot_max = dcdtimes - dt_snapshots.at(cindex); //maximum initial snapshot
        //int nt0s_max = (t0_snapshot_max - t0_snapshot_sm) / t0_snapshot_dif;
        //double s41 = 0;
        int t0_index = 0;
        for(int t0_snapshot = t0_snapshot_sm; t0_snapshot < t0_snapshot_max; t0_snapshot += t0_snapshot_dif) {

        // ========================================================================== //
        //      final snapshot

            int tf_snapshot = t0_snapshot + dt_snapshots.at(cindex); //calculate final snapshot
            whichfile = tf_snapshot / filesnapshots;
            strcpy(filename, filenames_str[whichfile].c_str());
            /*
            //save final snapshot info to meta file
	        metafile << filename << endl;
	        metafile << "final snapshot " << tf_snapshot << endl;
            */
            //read final position data
            gdcdp(xf, yf, zf, filename, (tf_snapshot - (whichfile * filesnapshots)), flag, &pos, cellsize);

            memset(&filename, 0, sizeof(filesnapshots));
            ///find final snapshot CM

            if(tf_snapshot > filesnapshots * (whichfile+1)) cout << "ERROR: tf_snapshot outside file: tf_snapshot = " << tf_snapshot <<  " max snapshot = " << filesnapshots * (whichfile+1) << endl;

            if(cm_times.at(tf_snapshot) == false) {
                cm_x.at(tf_snapshot) = cm(xf, N);
                cm_y.at(tf_snapshot) = cm(yf, N);
                cm_z.at(tf_snapshot) = cm(zf, N);
                cm_times.at(tf_snapshot) == true;
            }

        // ========================================================================== //
        //      initial snapshot
            whichfile = t0_snapshot / filesnapshots;
            strcpy(filename, filenames_str[whichfile].c_str());
            /*
            //save initial snapshot info to meta file
            metafile << filename << endl;
            metafile << "initial snapshot " << t0_snapshot << endl;
            */
            //read initial position data
            gdcdp(x0, y0, z0, filename, (t0_snapshot - (whichfile * filesnapshots)),flag, &pos, cellsize);

            memset(&filename, 0, sizeof(filesnapshots));
            //find initial snapshot CM
            if(cm_times.at(t0_snapshot) == false) {
                cm_x.at(t0_snapshot) = cm(x0, N);
                cm_y.at(t0_snapshot) = cm(y0, N);
                cm_z.at(t0_snapshot) = cm(z0, N);
                cm_times.at(t0_snapshot) = true;
            }
            //save particle index to meta file
            /*
            if(ntints==1) {
            metafile << "particle index " << N-1 << endl;
            }
            */
        // ========================================================================== //
        //      calculate function values
            //int t0_index = t0_snapshot - t0_snapshot_sm;
            //cout << "t0 = " << t0_snapshot*dtsave << endl;
            //N=2;
            for(int i = 0; i < N; i++) { //loop over particles
            //cout << "particle " << i << endl;
                //displacements
                dx = min(abs((xf[i] - cm_x.at(tf_snapshot)) - (x0[i] - cm_x.at(t0_snapshot))),L-abs((xf[i] - cm_x.at(tf_snapshot)) - (x0[i] - cm_x.at(t0_snapshot)))); //enlarge to size tint_lg/dtsave and use tf_snapshot, t0_snapshot as indices
                dy = min(abs((yf[i] - cm_y.at(tf_snapshot)) - (y0[i] - cm_y.at(t0_snapshot))),L-abs((yf[i] - cm_y.at(tf_snapshot)) - (y0[i] - cm_y.at(t0_snapshot))));
                dz = min(abs((zf[i] - cm_z.at(tf_snapshot)) - (z0[i] - cm_z.at(t0_snapshot))),L-abs((zf[i] - cm_z.at(tf_snapshot)) - (z0[i] - cm_z.at(t0_snapshot))));
                //self-intermediate scattering function
                fs_x.at(cindex) += cos(q * dx);
                fs_y.at(cindex) += cos(q * dy);
                fs_z.at(cindex) += cos(q * dz);

                //mean square displacement
                double dr2 = dx*dx + dy*dy + dz*dz;
                msd.at(cindex) += dr2;
                //overlap and s4
                complex<double> wadd;
                complex<double> q_dot_r;
                if(a2 > dr2) {
                    //cout << "overlap=1" << endl;
                    //cout << "x=" << x0[i] << " y=" << y0[i] << " z=" << z0[i] << endl;
                    fo.at(cindex) += 1;
                    /*
                    for(int j = 0; j < nqs; j++) { //loop over qs
                        q_dot_r = {0,-qx_vect.at(j)*x0[i] - qy_vect.at(j)*y0[i] - qz_vect.at(j)*z0[i]};
                        //if(j==nqs-1) cout << "q*r=" << q_dot_r << endl;
                        //if(qmags.at(j)<0.0000000001) wadd={1,0};
                        wadd = exp(q_dot_r);
                        if(t0_index==1 && j==1){
                            //cout << "qdotr" << q_dot_r << endl;
                        }
                        w.at(j).at(t0_index) += wadd;
                        //w2.at(j).at(t0_index) += norm(wadd); //always 1???
                        if(j==nqs-1) {
                            //cout << "norm(wadd)=" << norm(wadd) << endl;
                            //cout << "wadd" << wadd << endl;
                            //cout << "w2=" << w2.at(j).at(t0_index) << endl;
                            //cout << "w[re]=" << wadd.real() << endl;
                            //cout << "w[im]=" << wadd.imag() << endl;
                        }
                    }
                    */
                }
            }
            //cout << "w[re]_sum=" << w.at(nqs-1).at(t0_index).real() << endl;
            //cout << "w[im]_sum=" << w.at(nqs-1).at(t0_index).imag() << endl;
            //w2.at(j).at(t0_index)

            normalization.at(cindex) += N; //increment main normalization counter
            /*
            for(int j = 0; j < nqs; j++) { //loop over qs
                w_avg.at(j) += w.at(j).at(t0_index);
                w2_avg.at(j) += norm(w.at(j).at(t0_index));
                w.at(j).at(t0_index) = {0,0};
                //if(j==nqs-1) cout << "w2_avg=" << w2_avg.at(j) << endl;
            }
            */
            nt0s.at(cindex)++;
            t0_index++;
        }
        //cout << "w2_avg= " << w2_avg.at(nqs-1)/nt0s << endl;
        //cout << "fo= " << fo.at(cindex);
        //fill s4
        /*
        for(int j = 0; j < nqs; j++) {
            s4.at(j).at(cindex) = w2_avg.at(j)/nt0s.at(cindex)/N;
            w2_avg.at(j) = 0;
            w_avg.at(j) = 0;
            //cout << "s4(" << j  << ", " << dt_snapshots.at(cindex)*dtsave << ")" << s4.at(j).at(cindex)/normalization.at(cindex) << endl;
        }
        */
    }
    //subtract off f0 for q=0
    /*
    for(int tindex = 0; tindex < ntints; tindex++) {
        double fo_double = fo.at(tindex);
        int nt0s_t = nt0s.at(tindex);
        s4.at(0).at(tindex) -= fo_double*fo_double/nt0s_t/nt0s_t/N;
    }
    */

// ========================================================================== //
// Normalize & Print
// ========================================================================== //

    cout << "Normalizing & printing..." << endl;
    ofstream outfile("output.dat"); //open output files
    //ofstream s4file("s4.dat");
    /*
    s4file << fixed << setprecision(8) << "time | q" << ",";
    for(int j=0; j<nqs; j++) {
            s4file << fixed << setprecision(8) << qmags.at(j);
            if(j < nqs-1) s4file << ",";
        }
    s4file << endl;
    */
    //normalize and print functions to output file
    for(int tindex = 0; tindex < ntints; tindex++) {
        msd.at(tindex) /= (normalization.at(tindex));
        fs_x.at(tindex) /= (normalization.at(tindex));
        fs_y.at(tindex) /= (normalization.at(tindex));
        fs_z.at(tindex) /= (normalization.at(tindex));
        fo.at(tindex) /= (normalization.at(tindex));
        //s4.at(tindex) /= (norm.at(tindex));

        outfile << dt_snapshots.at(tindex)*dtsave*delta_t << ",";
        outfile << fixed << setprecision(8) << msd.at(tindex) << ",";
        outfile << fixed << setprecision(8) << fo.at(tindex) << ",";
        outfile << fixed << setprecision(8) << fs_x.at(tindex) << ",";
        outfile << fixed << setprecision(8) << fs_y.at(tindex) << ",";
        outfile << fixed << setprecision(8) << fs_z.at(tindex);
        /*
        s4file << dt_snapshots.at(tindex)*dtsave << ",";
        for(int j=0; j<nqs; j++) {
            s4file << fixed << setprecision(8) << s4.at(j).at(tindex);
            if(j < nqs-1) s4file << ",";
        }
        */
        if(tindex < dt_snapshots.size() - 1) {
            outfile << endl;
            //s4file << endl;
        }
    }
    outfile.close();

    cout << "DONE" << endl;
    time_t t2 = time(0);
    cout << "Run time: " << t2-t1 << "s" << endl;
    //https://stackoverflow.com/questions/24840205/how-to-print-running-time-of-a-case-in-c
    return 0;
}

void sortqs(vector<double> &qx, vector<double> &qy, vector<double> &qz, vector<int> itable) {
    int size = qx.size();
    vector<double> qxt = qx;
    vector<double> qyt = qy;
    vector<double> qzt = qz;
    for(int i = 0; i < size; i++) {
        //if(itable.at(i) != i) {
            qx.at(i) = qxt.at(itable.at(i));
            qy.at(i) = qyt.at(itable.at(i));
            qz.at(i) = qzt.at(itable.at(i));
        //}
    }
}

float cm(float *x, int N) {
    float cm = 0.0;
    for(int i = 0; i < N; i++) {
        cm += x[i];
    }
    cm /= N;
    return cm;
}

void printParams(string path, int nfiles, int tfirst, int tint_sm, int tint_lg, int dtsave, int ntints, char t0_choice, int t0_dif, float q, float a, double qmax, double dt) {

    ofstream file;
    string file_path = __FILE__;
    //string dir_path = file_path.substr(0, file_path.rfind("\\"));
    //https://stackoverflow.com/questions/26127075/how-to-return-the-directory-of-the-cpp-file
    file.open("params.txt");
    file << "Run date: " << __DATE__ << " , " << __TIME__ << endl;
    file << "Data Directory: " << path << endl;
    file << "Code Directory: " << file_path << endl << endl;
    file << "Number of files: " << nfiles << endl;
    file << "TIME AVERAGING" << endl;
    file << "First initial time used for averaging: " << tfirst << endl;
    file << "Smallest time interval: " << tint_sm << endl;
    file << "Largest time interval: " << tint_lg << endl;
    file << "Gap between recorded times: " << dtsave << endl;
    file << "Total number of time intervals: " << ntints << endl;
    file << "Set value for spacing between initial times: ";
    if(t0_choice == 'y' || t0_choice == 'Y') {
        file << "YES, spacing = " << t0_dif << endl << endl;
    }
    else file << "NO" << endl << endl;
    file << "PARAMETERS" << endl;
    file << "q = " << q << endl;
    file << "a = " << a;
    file << "qmax = " << qmax;
    file << "dt = " << dt;
    file.close();
}