/* File:     fractal.cpp
 *
 * Purpose:  compute the Julia set fractals
 *
 * Compile:  g++ -g -Wall -fopenmp -o fractal fractal.cpp -lglut -lGL
 * Run:      ./fractal
 *
 */

#include <iostream>
#include <cstdlib>
#include "../common/cpu_bitmap.h"
#include <omp.h>
using namespace std;

#define DIM 768
/*Uncomment the following line for visualization of the bitmap*/
//#define DISPLAY 1

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    //cuComplex c(-0.8, 0.156);
    cuComplex c(-0.7269, 0.1889);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<300; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

/*Parallelize the following function using OpenMP*/
// void kernel_omp ( unsigned char *ptr ){
//     for (int y=0; y<DIM; y++) {
//         for (int x=0; x<DIM; x++) {
//             int offset = x + y * DIM;

//             int juliaValue = julia( x, y );
//             ptr[offset*4 + 0] = 255 * juliaValue;
//             ptr[offset*4 + 1] = 0;
//             ptr[offset*4 + 2] = 0;
//             ptr[offset*4 + 3] = 255;
//         }
//     }
//  }

//1D Row parallelization
void kernel_omp_1D_Row(unsigned char *ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        for (int y = thread_id; y < DIM; y += num_threads) {
            for (int x = 0; x < DIM; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}

//1D Column parallelization
void kernel_omp_1D_Col(unsigned char *ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        for (int x = thread_id; x < DIM; x += num_threads) {
            for (int y = 0; y < DIM; y++) {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}

//2D Row-Block parallelization
void kernel_omp_2D_Row_Block(unsigned char *ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        int rows_per_thread = DIM / num_threads;
        
        int start_row = thread_id * rows_per_thread;
        
        int end_row;
        if (thread_id == num_threads - 1) {
            // Last thread takes any remaining rows
            end_row = DIM;
        } else {
            end_row = start_row + rows_per_thread;
        }
        
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < DIM; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}

//2D Col-Block parallelization
void kernel_omp_2D_Col_Block(unsigned char *ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        int cols_per_thread = DIM / num_threads;
        
        int start_col = thread_id * cols_per_thread;
        
        int end_col;
        if (thread_id == num_threads - 1) {
            // Last thread takes any remaining columns
            end_col = DIM;
        } else {
            end_col = start_col + cols_per_thread;
        }
        
        for (int y = 0; y < DIM; y++) {
            for (int x = start_col; x < end_col; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia(x, y);
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}

//OpenMP for
void kernel_omp_For(unsigned char *ptr) {

    #pragma omp parallel for //collapse(2)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            int juliaValue = julia(x, y);
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}
 
 void kernel_serial ( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr_s = bitmap.get_ptr();
    unsigned char *ptr_p = bitmap.get_ptr(); 
    double start, finish_s, finish_p, finish_1d_row, finish_1d_col, 
    finish_2d_row_block, finish_2d_col_block, finish_omp_for; 
    
    /*Serial run*/
    start = omp_get_wtime();
    kernel_serial( ptr_s );
	finish_s = omp_get_wtime() - start;
    
    /*Parallel run*/ 
    // start = omp_get_wtime();
    // kernel_omp( ptr_p );
	// finish_p = omp_get_wtime() - start;

    /*Parallel 1D row-wise parallelization*/
    start = omp_get_wtime();
    kernel_omp_1D_Row( ptr_p );
    finish_1d_row = omp_get_wtime() - start;

    /*Parallel 1D col-wise parallelization*/
    start = omp_get_wtime();
    kernel_omp_1D_Col( ptr_p );
    finish_1d_col = omp_get_wtime() - start;

    /*Parallel 2D row-block parallelization*/
    start = omp_get_wtime();
    kernel_omp_2D_Row_Block( ptr_p );
    finish_2d_row_block = omp_get_wtime() - start;

    /*Parallel 2D col-block parallelization*/
    start = omp_get_wtime();
    kernel_omp_2D_Col_Block( ptr_p );
    finish_2d_col_block = omp_get_wtime() - start;

    //OpenMP for
    start = omp_get_wtime();
    kernel_omp_For( ptr_p );
    finish_omp_for = omp_get_wtime() - start;
    
    cout << "Elapsed time: " << endl;
    cout << "Serial time: " << finish_s << endl;
    //cout << "Parallel time: " << finish_p << endl;
    cout << "1D Row Parallel time: " << finish_1d_row << endl;
    cout << "1D Column Parallel time: " << finish_1d_col << endl;
    cout << "2D Row-Block Parallel time: " << finish_2d_row_block << endl;
    cout << "2D Col-Block Parallel time: " << finish_2d_col_block << endl;
    cout << "OpenMP for Parallel time: " << finish_omp_for << endl;
    cout << "" << endl;

    //cout << "Speedup: " << finish_s/finish_p << endl;
    cout << "1D Rowwise Speedup: " << finish_s/finish_1d_row << endl;
    cout << "1D Column Speedup: " << finish_s/finish_1d_col << endl;
    cout << "2D Row-Block Speedup: " << finish_s/finish_2d_row_block << endl;
    cout << "2D Col-Block Speedup: " << finish_s/finish_2d_col_block << endl;
    cout << "OpenMP for Speedup: " << finish_s/finish_omp_for << endl;
	    
    #ifdef DISPLAY     
    bitmap.display_and_exit();
    #endif
    return 0;
}
