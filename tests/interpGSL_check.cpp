#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>

#include <string>
#include <vector>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include <chrono>

using namespace std;

std::vector<gsl_spline*> temperature_spline_cache;
std::vector<gsl_interp_accel*> temperature_accel_cache;
std::vector<gsl_spline*> pressure_spline_cache;
std::vector<gsl_interp_accel*> pressure_accel_cache;
std::vector<gsl_spline*> cs2_spline_cache;
std::vector<gsl_interp_accel*> cs2_accel_cache;
std::vector<gsl_spline*> entropy_spline_cache;
std::vector<gsl_interp_accel*> entropy_accel_cache;

double interpolate_spline(double e,  vector<double>& xe, vector<double>& ye, const std::vector<gsl_spline*>& spline_cache, 
    const std::vector<gsl_interp_accel*>& accel_cache) {

    const int N_e = xe.size();
    
    if (e >= xe[0] && e < xe[N_e - 1]) {
        return gsl_spline_eval(spline_cache[0], e, accel_cache[0]);
    }
    
    if (e < xe[0]) {
        const double slope = (ye[1] - ye[0]) / (xe[1] - xe[0]);
        return ye[0] + slope * (e - xe[0]);
    } else {
        const double slope = (ye[N_e - 1] - ye[N_e - 2]) / (xe[N_e - 1] - xe[N_e - 2]);
        return ye[N_e - 2] + slope * (e - xe[N_e - 2]);
    }
}


int main() {

     auto start = std::chrono::steady_clock::now();

    std::ifstream eos_file("/home/luizaperin/masters_research/gp/data_files/constrained/l10_s15/eos_w_s0_l10_s15_clean.dat");
    if (!eos_file.is_open()) {
        std::cerr << "Error: Could not open EOS file." << std::endl;
        return 1;
    }

    std::string line;

    int e_length;
    eos_file >> e_length;
    std::cout << e_length << std::endl;

    vector<double> energy_tb(e_length);
    vector<double> temperature_tb(e_length);
    vector<double> pressure_tb(e_length);   
    vector<double> cs2_tb(e_length);
    vector<double> entropy_tb(e_length); 


    std::getline(eos_file, line);

    int ii = 0;

    int N_e = e_length;

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

    temperature_spline_cache.push_back(gsl_spline_alloc(gsl_interp_linear, N_e));
    temperature_accel_cache.push_back(gsl_interp_accel_alloc());
    gsl_spline_init(temperature_spline_cache[0], energy_tb.data(), temperature_tb.data(), N_e);
        
    // Pressure spline
    pressure_spline_cache.push_back(gsl_spline_alloc(gsl_interp_linear, N_e));
    pressure_accel_cache.push_back(gsl_interp_accel_alloc());
    gsl_spline_init(pressure_spline_cache[0], energy_tb.data(), pressure_tb.data(), N_e);
        
    // cs2 spline
    cs2_spline_cache.push_back(gsl_spline_alloc(gsl_interp_linear, N_e));
    cs2_accel_cache.push_back(gsl_interp_accel_alloc());
    gsl_spline_init(cs2_spline_cache[0], energy_tb.data(), cs2_tb.data(), N_e);
        
    // Entropy spline
    entropy_spline_cache.push_back(gsl_spline_alloc(gsl_interp_linear, N_e));
    entropy_accel_cache.push_back(gsl_interp_accel_alloc());
    gsl_spline_init(entropy_spline_cache[0], energy_tb.data(), entropy_tb.data(), N_e);

    if (ii < e_length) {
        std::cerr << "Error: EOS file contains only " << ii << " valid lines, expected " << e_length << "." << std::endl;
        return 1;
    }

    std::ofstream fout("./gp/tests/interpGSL_check.dat");

    // --- 1) check exact table points
    //fout << "# index e_val T_tab T_int diff\n";

/*      for (int i = 0; i < N_e; ++i) {
        double e_val  = energy_tb[i];
        double table_tab  = temperature_tb[i];
        double table_int  = interpolate_spline(e_val,
                                                     energy_tb, temperature_tb, temperature_spline_cache, temperature_accel_cache);

        fout << i << " " << e_val << " " << table_tab << " " << table_int
             << " " << (table_int - table_tab) << "\n";
    }  */

    fout << "\n# random probes\n";
    double e_min = energy_tb[0];
    double e_max = energy_tb[N_e-1];
    for (int k = 0; k < 100; ++k) {
        double e_rand = e_min + (e_max - e_min) * (double(rand()) / RAND_MAX);
        double T_int  = interpolate_spline(e_rand,
                                                 energy_tb, temperature_tb, temperature_spline_cache, temperature_accel_cache);

    fout << k << " " << e_rand << " " << T_int << "\n";
    }

    auto end = std::chrono::steady_clock::now();

    // 3. Calculate the duration and cast it to the desired unit (e.g., milliseconds)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 4. Print the result
    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;


}