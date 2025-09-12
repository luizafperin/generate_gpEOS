#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
using namespace std;


double interpolate1D_nonuniform(double e, vector<double> table_x, vector<double> table_y){
    int N_e = 7001;;
    const double tol = 1e-12;

    // --- underflow: extrapolate with first two points
    if (e < table_x[0]) {
        double slope = (table_y[1] - table_y[0]) / (table_x[1] - table_x[0]);

        double result_low = table_y[0] + slope * (e - table_x[0]);

        if (!std::isfinite(result_low)) {
        std::cerr << "⚠️ interpolate1D_nonuniform produced NaN/Inf at e=" << e;
    }
        return result_low;
    }

    // --- overflow: extrapolate with last two points
    if (e >= table_x[N_e-1]) {
        double slope = (table_y[N_e-1] - table_y[N_e-2]) / (table_x[N_e-1] - table_x[N_e-2]);
        return table_y[N_e-2] + slope * (e - table_x[N_e-2]);
    }

    // --- exact match check (within tolerance)
    for (int j = 0; j < N_e; ++j) {
        if (fabs(e - table_x[j]) < tol) {
            return table_y[j];
        }
    }

    // --- binary search to find i such that xe[i] <= e < xe[i+1]
    int low = 0, high = N_e - 1;
    while (high - low > 1) {
        int mid = (low + high) / 2;
        if (table_x[mid] > e) {
            high = mid;
        } else {
            low = mid;
        }
    }
    int i = low;

    // --- linear interpolation
    double denominator = table_x[i+1] - table_x[i];

    if (fabs(denominator) < tol) {
        return 0.5 * (table_y[i] + table_y[i+1]); // instead of interpolating, it just returns the average
    }

    double frac = (e - table_x[i]) / denominator;
    double result = table_y[i] * (1.0 - frac) + table_y[i+1] * frac;

    // --- safety check
    if (!std::isfinite(result)) {
        std::cerr << "⚠️ interpolate1D_nonuniform produced NaN/Inf at e=" << e
                  << " between xe[" << i << "]=" << table_x[i]
                  << " and xe[" << i+1 << "]=" << table_x[i+1] << std::endl;
    }

    return result;
}

int main() {
    int N_e  = 7001;

    std::ifstream eos_file("/home/luizaperin/MUSIC-3.1/EOS/EOS-gp/eos_weighted.dat");
    if (!eos_file.is_open()) {
        std::cerr << "Error: Could not open EOS file." << std::endl;
        return 1;
    }

    std::string line;
        
    int e_length = N_e;
    vector<double> energy_tb(e_length);
    vector<double> temperature_tb(e_length);
    vector<double> pressure_tb(e_length);   
    vector<double> cs2_tb(e_length);
    vector<double> entropy_tb(e_length);

    int ii = 0;
        
    while (getline(eos_file, line) && ii < e_length) {
        std::istringstream iss(line);
        double energy, temperature, pressure, cs2, entropy;
        if (!(iss >> energy >> temperature >> pressure >> cs2 >> entropy)) {
            std::cerr << "Error: Malformed line in EOS file at line " << ii + 1 << std::endl;
            return 1;
        }

        temperature_tb[ii] = temperature;    
        pressure_tb[ii] = pressure; // 
        cs2_tb[ii] = cs2; //dimensionless
        entropy_tb[ii] = entropy; 
        energy_tb[ii] = energy;

        ++ii;
    }

    if (ii < e_length) {
        std::cerr << "Error: EOS file contains only " << ii << " valid lines, expected " << e_length << "." << std::endl;
        return 1;
    }

    std::ofstream fout("interp_check.dat");

    // --- 1) check exact table points
    //fout << "# index e_val T_tab T_int diff\n";

/*     for (int i = 0; i < N_e; ++i) {
        double e_val  = energy_tb[i];
        double cs2_tab  = entropy_tb[i];
        double cs2_int  = interpolate1D_nonuniform(e_val,
                                                     energy_tb, entropy_tb);

        fout << i << " " << e_val << " " << cs2_tab << " " << cs2_int
             << " " << (cs2_int - cs2_tab) << "\n";
    } */

// --- 2) check midpoints between grid values

/* fout << "\n# midpoints\n";
for (int i = 0; i < N_e - 1; ++i) {
    double e_mid = 0.5 * (energy_tb[i] +
                          energy_tb[i+1]);
    double T_int = interpolate1D_nonuniform(e_mid, energy_tb, temperature_tb);

    //fout << "mid " << i << " " << e_mid << " " << T_int << "\n";
}  */

// --- 3) random probes in full range
    fout << "\n# random probes\n";
    double e_min = energy_tb[0];
    double e_max = energy_tb[N_e-1];
    for (int k = 0; k < 100; ++k) {
        double e_rand = e_min + (e_max - e_min) * (double(rand()) / RAND_MAX);
        double T_int  = interpolate1D_nonuniform(e_rand,
                                                 energy_tb, temperature_tb);

    fout << k << " " << e_rand << " " << T_int << "\n";
    }

    fout.close();
    std::cout << "Wrote interp_check.dat with random tests.\n";
    return 0;



}

