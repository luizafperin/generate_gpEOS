#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <chrono>

using namespace std;


double interpolate1D_nonuniform(double e, vector<double>& xe, vector<double>& ye){
    int N_e = xe.size();
    
    // --- underflow: extrapolate with first two points
    if (e < xe[0]) {
        const double slope = (ye[1] - ye[0]) / (xe[1] - xe[0]);
        return ye[0] + slope * (e - xe[0]);
    }

    // --- overflow: extrapolate with last two points
    if (e >= xe[N_e - 1]) {
        const double slope = (ye[N_e - 1] - ye[N_e - 2]) / (xe[N_e - 1] - xe[N_e - 2]);
        return ye[N_e - 2] + slope * (e - xe[N_e - 2]);
    }

    // --- binary search using iterators
    auto it = std::upper_bound(xe.begin(), xe.end(), e);  // first xe[k] > e
    int i = std::max(0, static_cast<int>(it - xe.begin()) - 1);

    // --- linear interpolation
    const double dx = xe[i + 1] - xe[i];

    const double tol = 1e-12;
    if (fabs(dx) < tol) {
        cout << "Warning: interpolate1D_nonuniform found two identical xe values at index "
                  << i << ": xe[" << i << "] = xe[" << i + 1 << "] = " << xe[i] << std::endl;
        return 0.5 * (ye[i] + ye[i+1]);
    }

    const double frac = (e - xe[i]) / dx;
    const double result = ye[i] * (1.0 - frac) + ye[i + 1] * frac;

    if (!std::isfinite(result)) {
        std::cerr << "interpolate1D_nonuniform produced NaN/Inf at e=" << e
                  << " between xe[" << i << "]=" << xe[i]
                  << " and xe[" << i + 1 << "]=" << xe[i + 1] << std::endl;
    }
    return result;
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

    if (ii < e_length) {
        std::cerr << "Error: EOS file contains only " << ii << " valid lines, expected " << e_length << "." << std::endl;
        return 1;
    }

    std::ofstream fout("./gp/tests/interp_check.dat");

    // --- 1) check exact table points
    //fout << "# index e_val T_tab T_int diff\n";

/*      for (int i = 0; i < N_e; ++i) {
        double e_val  = energy_tb[i];
        double table_tab  = temperature_tb[i];
        double table_int  = interpolate1D_nonuniform(e_val,
                                                     energy_tb, temperature_tb);

        fout << i << " " << e_val << " " << table_tab << " " << table_int
             << " " << (table_int - table_tab) << "\n";
    } 
 */
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
    //std::cout << "Wrote interp_check.dat with random tests.\n";

    auto end = std::chrono::steady_clock::now();

    // 3. Calculate the duration and cast it to the desired unit (e.g., milliseconds)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 4. Print the result
    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;


}

