/*
 * nanoBraggCUDA.cu -- production single-precision (float) CUDA kernel computing
 * nanoBragg diffraction images. All device arithmetic is float.
 *
 * Entry point: extern "C" nanoBraggSpotsCUDA(...), which marshals the host
 * (double) parameters to the device, launches nanoBraggSpotsCUDAKernel,
 * and reduces the result image.
 *
 * Two structural facts a reader needs:
 *   - The per-(phi, mosaic) rotated unit-cell vectors are precomputed on the
 *     host (in double, matching the nanoBraggCPU.c reference) and read from the
 *     phi_mos_* tables in the hot loop, so the kernel does no in-loop rotation.
 *   - The lattice shape transform uses a delta-reduced sincg: the fractional
 *     Miller index is reduced to |delta| <= 0.5 before the trig call.
 */

/* Configuration and types */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <driver_types.h>
#include "nanotypes.h"
#include "nanoBraggCUDA.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define VECTOR_SIZE 4

struct structureFactorParams {
    short hkls;
    short h_min;
    short h_max;
    short h_range;
    short k_min;
    short k_max;
    short k_range;
    short l_min;
    short l_max;
    short l_range;
};

struct constParams {
    float Avogadro;
    float r_e_sqr;
};

struct beamSource {
    float neg_unit_source_vector[VECTOR_SIZE];
    float intensity;
    float lambda;
};

struct beamParams {
    float beam_vector[VECTOR_SIZE];
    float fluence;
    bool calc_polar;
    float polarization;
    float polar_vector[VECTOR_SIZE];
    short sources;
};

struct detectorParams {
    short spixels;
    short fpixels;
    short roi_xmin;
    short roi_xmax;
    short roi_ymin;
    short roi_ymax;
    short oversample;
    bool point_pixel;
    float pixel_size;
    float subpixel_size;
    long steps;
    float detector_thickstep;
    short detector_thicksteps;
    float detector_thick;
    float detector_mu;
    bool curved_detector;
    float sdet_vector[VECTOR_SIZE];
    float fdet_vector[VECTOR_SIZE];
    float odet_vector[VECTOR_SIZE];
    float pix0_vector[VECTOR_SIZE];
};

struct sampleParams {
    float distance;
    float close_distance;
    float water_size;
    float water_F;
    float water_MW;
};

struct unitCell {
    float a0[VECTOR_SIZE];
    float b0[VECTOR_SIZE];
    float c0[VECTOR_SIZE];
    float V_cell;
};

struct crystalParams {
    unitCell uc;
    float Na;
    float Nb;
    float Nc;
    float default_F;
    float dmin;
    shapetype xtal_shape;
    short mosaic_domains;
    structureFactorParams fhklParams;
};

struct goniometerParams {
    float phi0;
    float phistep;
    short phisteps;
    float spindle_vector[VECTOR_SIZE];
};

/* Forward declarations */

__global__ void nanoBraggSpotsCUDAKernel(const detectorParams * __restrict__ detectorPtr, const beamParams * __restrict__ beamPtr, const goniometerParams * __restrict__ goniometerPtr, const sampleParams * __restrict__ samplePtr,
        const crystalParams * crystalPtr, const constParams * __restrict__ constantsPtr, const beamSource * __restrict__ beam_sources, const float * __restrict__ Fhkl,
        const float * __restrict__ phi_mos_a, const float * __restrict__ phi_mos_b, const float * __restrict__ phi_mos_c, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/, float * max_I_y_reduction /*out*/, bool * rangemap);

/* vector inner product where vector magnitude is 0th element */
__device__ __inline__ static float dot_product(const float * x, const float * y);
/* vector cross product where vector magnitude is 0th element */
__device__ static float *cross_product(const float * x, const float * y, float * z);
__device__ __inline__ float norm3d_fma_rn(float v1, float v2, float v3);
/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
__device__ static float unitize(float * vector,
float *new_unit_vector);
/* rotate a 3-vector about a unit vector axis */
__device__ static float *rotate_axis(const float * __restrict__ v,
float *newv, const float * __restrict__ axis, const float phi);
__device__ __inline__ static long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range);
__device__ __inline__ float quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const float * __restrict__ Fhkl);
/* load the fully phi+mosaic-rotated cell (a/b/c[4], element[0]=0) from the host
   phi_mos_* table via __ldg; arguments are (table, out, index). */
__device__ __forceinline__ void load_rotated_cell_ldg(const float * __restrict__ tbl, float out[4], int idx_base);
/* nearest_hkl: integer Miller index of the nearest reciprocal-lattice point
   (the Fhkl structure-factor lookup index) */
__device__ __forceinline__ static short nearest_hkl(float h);
/* fractional: distance from the nearest Bragg peak, reduced to |delta| <= 0.5 */
__device__ __forceinline__ static float fractional(float h);
/* delta-reduced sincg: N-slit interference function evaluated from the already-reduced
   delta = h - rint(h); pi is applied inside */
__device__ __inline__ static float sincg_delta(float delta, float N);
/* Fourier transform of a sphere */
__device__ static float sinc3(float x);
/* polarization factor from vectors */
__device__ static float polarization_factor(float kahn_factor, const float * __restrict__ unitIncident, float *unitDiffracted,
        const float * __restrict__ unitAxis);

/* Host code */

/* Check the return value of the CUDA runtime API call and exit
   the application if the call has failed. */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

static void doubleVectorToFloatVector(float * dest, double * src, size_t vector_items) {
    for (size_t i = 0; i < vector_items; i++) {
        dest[i] = src[i];
    }
}

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
static double unitizeCPU(double * vector, double * new_unit_vector) {

    double v1 = vector[1];
    double v2 = vector[2];
    double v3 = vector[3];

    double mag = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

    if (mag != 0.0) {
        /* normalize it */
        new_unit_vector[0] = mag;
        new_unit_vector[1] = v1 / mag;
        new_unit_vector[2] = v2 / mag;
        new_unit_vector[3] = v3 / mag;
    } else {
        /* can't normalize, report zero vector */
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;
    }
    return mag;
}

extern "C" void nanoBraggSpotsCUDA(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
        double pixel_size, double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu,
        double sdet_vector[4], double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance,
        double beam_vector[4], double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources,
        double *source_X, double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4],
        shapetype xtal_shape, double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell,
        double water_size, double water_F, double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F,
        int interpolate, double *** Fhkl, int h_min, int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls,
        int nopolar, double polar_vector[4], double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/,
        double * omega_sum/*out*/, int * sumn /*out*/, double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/,
        double * max_I_y /*out*/) {

    int total_pixels = spixels * fpixels;

    bool * rangemap = (bool*) calloc(total_pixels, sizeof(bool));
    float * omega_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_x_reduction = (float*) calloc(total_pixels, sizeof(float));
    float * max_I_y_reduction = (float*) calloc(total_pixels, sizeof(float));

    memset(floatimage, 0, sizeof(typeof(*floatimage)) * total_pixels);

    detectorParams detector;
    detector.spixels = spixels;
    detector.fpixels = fpixels;
    detector.roi_xmin = roi_xmin;
    detector.roi_xmax = roi_xmax;
    detector.roi_ymin = roi_ymin;
    detector.roi_ymax = roi_ymax;
    detector.oversample = oversample;
    detector.point_pixel = point_pixel;
    detector.pixel_size = pixel_size;
    detector.subpixel_size = (float) subpixel_size;
    detector.steps = steps;
    detector.detector_thickstep = detector_thickstep;
    detector.detector_thicksteps = detector_thicksteps;
    detector.detector_thick = detector_thick;
    detector.detector_mu = detector_mu;
    detector.curved_detector = curved_detector;
    doubleVectorToFloatVector(detector.sdet_vector, sdet_vector, VECTOR_SIZE);
    doubleVectorToFloatVector(detector.fdet_vector, fdet_vector, VECTOR_SIZE);
    doubleVectorToFloatVector(detector.odet_vector, odet_vector, VECTOR_SIZE);
    doubleVectorToFloatVector(detector.pix0_vector, pix0_vector, VECTOR_SIZE);

    if (getenv("NB_DUMP_PIX0")) {
        fprintf(stderr, "NB_DUMP_PIX0 subpixel_size=%.17g pix0_vector=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) subpixel_size, (double) pix0_vector[0], (double) pix0_vector[1],
                (double) pix0_vector[2], (double) pix0_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 sdet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                sdet_vector[0], sdet_vector[1], sdet_vector[2], sdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 fdet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                fdet_vector[0], fdet_vector[1], fdet_vector[2], fdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 odet_vector(host double)=[%.17g, %.17g, %.17g, %.17g]\n",
                odet_vector[0], odet_vector[1], odet_vector[2], odet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 detector.sdet_vector(device float)=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) detector.sdet_vector[0], (double) detector.sdet_vector[1], (double) detector.sdet_vector[2], (double) detector.sdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 detector.fdet_vector(device float)=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) detector.fdet_vector[0], (double) detector.fdet_vector[1], (double) detector.fdet_vector[2], (double) detector.fdet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 detector.odet_vector(device float)=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) detector.odet_vector[0], (double) detector.odet_vector[1], (double) detector.odet_vector[2], (double) detector.odet_vector[3]);
        fprintf(stderr, "NB_DUMP_PIX0 detector.pix0_vector(device float)=[%.17g, %.17g, %.17g, %.17g]\n",
                (double) detector.pix0_vector[0], (double) detector.pix0_vector[1], (double) detector.pix0_vector[2], (double) detector.pix0_vector[3]);
    }

    beamParams beam;
    doubleVectorToFloatVector(beam.beam_vector, beam_vector, VECTOR_SIZE);
    beam.fluence = fluence;
    beam.calc_polar = !nopolar;
    beam.polarization = polarization;
    /* Unitize the polar vector once here on the host instead of once per pixel on the GPU. */
    double unit_polar_vector[VECTOR_SIZE];
    unitizeCPU(polar_vector, unit_polar_vector);
    doubleVectorToFloatVector(beam.polar_vector, unit_polar_vector, VECTOR_SIZE);
    beam.sources = sources;

    goniometerParams goniometer;
    goniometer.phi0 = phi0;
    goniometer.phistep = phistep;
    goniometer.phisteps = phisteps;
    doubleVectorToFloatVector(goniometer.spindle_vector, spindle_vector, VECTOR_SIZE);

    /* The unit-cell vectors fed to the h,k,l dot product are
         a/b/c = rotate_umat( rotate_axis(a0/b0/c0, spindle, phi), mosaic_umat[mos] )
       and depend only on (phi_tic, mos_tic); the spindle axis, base cell, phi steps and
       mosaic umats are all launch constants. Precompute the fully phi+mosaic-rotated cell
       once here on the host in double, using the same rotate_axis + rotate_umat algebra as
       the nanoBraggCPU.c reference, then cast each component to float. This keeps the
       rotation (and its sin/cos and matrix multiply) out of the per-subpixel/per-source hot
       loop, and matching the reference's double rotation improves parity. At phi==0 with no
       mosaic umat the rotation is an exact identity, so phi_mos_a[.] reproduces (float)a0[i]
       to the bit. Layout: phi_mos_a[(phi_tic*nmos + mos_tic)*3 + {0,1,2}] = rotated a's
       components {1,2,3}; same for b and c. */
    int nphi = phisteps > 0 ? phisteps : 1;
    int nmos = mosaic_domains > 0 ? mosaic_domains : 1;
    int pm_count = 3 * nphi * nmos;
    float * phi_mos_a_host = new float[pm_count];
    float * phi_mos_b_host = new float[pm_count];
    float * phi_mos_c_host = new float[pm_count];
    bool do_umat = (mosaic_spread > 0.0);
    for (int pt = 0; pt < nphi; pt++) {
        double phi_i = phistep * (double) pt + phi0;
        double sinphi = sin(phi_i);
        double cosphi = cos(phi_i);
        const double * srcv[3] = { a0, b0, c0 };
        double rot[3][4];   /* [vec][1..3] phi-rotated cell vector components, double */
        for (int _c = 0; _c < 3; ++_c) {
            const double * v = srcv[_c];
            /* rotate_axis algebra matching the nanoBraggCPU.c reference, in double */
            double dot = (spindle_vector[1] * v[1] + spindle_vector[2] * v[2] + spindle_vector[3] * v[3]) * (1.0 - cosphi);
            rot[_c][1] = spindle_vector[1] * dot + v[1] * cosphi + (-spindle_vector[3] * v[2] + spindle_vector[2] * v[3]) * sinphi;
            rot[_c][2] = spindle_vector[2] * dot + v[2] * cosphi + (+spindle_vector[3] * v[1] - spindle_vector[1] * v[3]) * sinphi;
            rot[_c][3] = spindle_vector[3] * dot + v[3] * cosphi + (-spindle_vector[2] * v[1] + spindle_vector[1] * v[2]) * sinphi;
        }
        float * dstv[3] = { phi_mos_a_host, phi_mos_b_host, phi_mos_c_host };
        for (int mt = 0; mt < nmos; mt++) {
            const double * um = mosaic_umats + (long) mt * 9; /* row-major double[9] */
            int base = (pt * nmos + mt) * 3;
            for (int _c = 0; _c < 3; ++_c) {
                double newv[4];
                if (do_umat) {
                    /* rotate_umat algebra matching the nanoBraggCPU.c reference, in double */
                    newv[1] = um[0] * rot[_c][1] + um[1] * rot[_c][2] + um[2] * rot[_c][3];
                    newv[2] = um[3] * rot[_c][1] + um[4] * rot[_c][2] + um[5] * rot[_c][3];
                    newv[3] = um[6] * rot[_c][1] + um[7] * rot[_c][2] + um[8] * rot[_c][3];
                } else {
                    newv[1] = rot[_c][1]; newv[2] = rot[_c][2]; newv[3] = rot[_c][3];
                }
                dstv[_c][base + 0] = (float) newv[1];
                dstv[_c][base + 1] = (float) newv[2];
                dstv[_c][base + 2] = (float) newv[3];
            }
        }
    }
    float * cu_phi_mos_a = NULL; float * cu_phi_mos_b = NULL; float * cu_phi_mos_c = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_a, sizeof(*cu_phi_mos_a) * pm_count));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_b, sizeof(*cu_phi_mos_b) * pm_count));
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_phi_mos_c, sizeof(*cu_phi_mos_c) * pm_count));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_a, phi_mos_a_host, sizeof(*cu_phi_mos_a) * pm_count, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_b, phi_mos_b_host, sizeof(*cu_phi_mos_b) * pm_count, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_phi_mos_c, phi_mos_c_host, sizeof(*cu_phi_mos_c) * pm_count, cudaMemcpyHostToDevice));

    sampleParams sample;
    sample.distance = distance;
    sample.close_distance = close_distance;
    sample.water_F = water_F;
    sample.water_size = water_size;
    sample.water_MW = water_MW;

    crystalParams crystal;
    crystal.default_F = default_F;
    crystal.dmin = dmin;
    crystal.Na = Na;
    crystal.Nb = Nb;
    crystal.Nc = Nc;
    crystal.uc.V_cell = V_cell;
    doubleVectorToFloatVector(crystal.uc.a0, a0, VECTOR_SIZE);
    doubleVectorToFloatVector(crystal.uc.b0, b0, VECTOR_SIZE);
    doubleVectorToFloatVector(crystal.uc.c0, c0, VECTOR_SIZE);
    crystal.xtal_shape = xtal_shape;
    crystal.fhklParams.hkls = hkls;
    crystal.fhklParams.h_max = h_max;
    crystal.fhklParams.h_min = h_min;
    crystal.fhklParams.h_range = h_range;
    crystal.fhklParams.k_max = k_max;
    crystal.fhklParams.k_min = k_min;
    crystal.fhklParams.k_range = k_range;
    crystal.fhklParams.l_max = l_max;
    crystal.fhklParams.l_min = l_min;
    crystal.fhklParams.l_range = l_range;
    crystal.mosaic_domains = mosaic_domains;

    /* Pad hkl with default_F value; */
    int h_min_pad = h_min - 1, h_max_pad = h_max + 1, h_range_pad = h_range + 2;
    int k_min_pad = k_min - 1, k_max_pad = k_max + 1, k_range_pad = k_range + 2;
    int l_min_pad = l_min - 1, l_max_pad = l_max + 1, l_range_pad = l_range + 2;
    int hklsize_pad = h_range_pad * k_range_pad * l_range_pad;
    float * FhklLinearPad = (float*) calloc(hklsize_pad, sizeof(*FhklLinearPad));
    for (int h = 0; h < h_range_pad; h++) {
        for (int k = 0; k < k_range_pad; k++) {
            for (int l = 0; l < l_range_pad; l++) {
                /* convert Fhkl double to float */
                if (h > 0 && h < h_range_pad - 1 && k > 0 && k < k_range_pad - 1 && l > 0 && l < l_range_pad - 1) {
                    FhklLinearPad[h * k_range_pad * l_range_pad + k * l_range_pad + l] = Fhkl[h - 1][k - 1][l - 1];
                } else {
                    FhklLinearPad[h * k_range_pad * l_range_pad + k * l_range_pad + l] = crystal.default_F;
                }
            }
        }
    }
    crystal.fhklParams.hkls = hkls;
    crystal.fhklParams.h_max = h_max_pad;
    crystal.fhklParams.h_min = h_min_pad;
    crystal.fhklParams.h_range = h_range_pad;
    crystal.fhklParams.k_max = k_max_pad;
    crystal.fhklParams.k_min = k_min_pad;
    crystal.fhklParams.k_range = k_range_pad;
    crystal.fhklParams.l_max = l_max_pad;
    crystal.fhklParams.l_min = l_min_pad;
    crystal.fhklParams.l_range = l_range_pad;

    constParams constants;
    constants.Avogadro = Avogadro;
    constants.r_e_sqr = r_e_sqr;

    detectorParams * cu_detector;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_detector, sizeof(*cu_detector)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_detector, &detector, sizeof(*cu_detector), cudaMemcpyHostToDevice));

    beamParams * cu_beam;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam, sizeof(*cu_beam)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_beam, &beam, sizeof(*cu_beam), cudaMemcpyHostToDevice));

    goniometerParams * cu_goniometer;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_goniometer, sizeof(*cu_goniometer)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_goniometer, &goniometer, sizeof(*cu_goniometer), cudaMemcpyHostToDevice));

    sampleParams * cu_sample;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_sample, sizeof(*cu_sample)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_sample, &sample, sizeof(*cu_sample), cudaMemcpyHostToDevice));

    crystalParams * cu_crystal;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_crystal, sizeof(*cu_crystal)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_crystal, &crystal, sizeof(*cu_crystal), cudaMemcpyHostToDevice));

    constParams * cu_constants;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_constants, sizeof(*cu_constants)));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_constants, &constants, sizeof(*cu_constants), cudaMemcpyHostToDevice));

    /* Repackage beam sources. Unitize source vectors and pack them together contiguously. */
    beamSource * beam_sources = new beamSource[beam.sources];
    for (int i = 0; i < beam.sources; i++) {
        double unitSource[VECTOR_SIZE] = { 0.0, -source_X[i], -source_Y[i], -source_Z[i] };
        unitizeCPU(unitSource, unitSource);
        doubleVectorToFloatVector(beam_sources[i].neg_unit_source_vector, unitSource, VECTOR_SIZE);
        beam_sources[i].lambda = source_lambda[i];
        beam_sources[i].intensity = source_I[i];
    }
    beamSource * cu_beam_sources = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_beam_sources, sizeof(*cu_beam_sources) * beam.sources));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_beam_sources, beam_sources, sizeof(*cu_beam_sources) * beam.sources, cudaMemcpyHostToDevice));

    float * cu_Fhkl = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_Fhkl, sizeof(*cu_Fhkl) * hklsize_pad));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_Fhkl, FhklLinearPad, sizeof(*cu_Fhkl) * hklsize_pad, cudaMemcpyHostToDevice));

    int unsigned short * cu_maskimage = NULL;
    if (maskimage != NULL) {
        CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_maskimage, sizeof(*cu_maskimage) * total_pixels));
        CUDA_CHECK_RETURN(cudaMemcpy(cu_maskimage, maskimage, sizeof(*cu_maskimage) * total_pixels, cudaMemcpyHostToDevice));
    }

    float * cu_floatimage = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_floatimage, sizeof(*cu_floatimage) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_floatimage, floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_omega_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_omega_reduction, omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_max_I_x_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_x_reduction, max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyHostToDevice));

    float * cu_max_I_y_reduction = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_max_I_y_reduction, max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyHostToDevice));

    bool * cu_rangemap = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void ** )&cu_rangemap, sizeof(*cu_rangemap) * total_pixels));
    CUDA_CHECK_RETURN(cudaMemcpy(cu_rangemap, rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyHostToDevice));

    int deviceId = 0;
    CUDA_CHECK_RETURN(cudaGetDevice(&deviceId));
    int smCount = 0;
    CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks(smCount * 32, 1);

    /* Time the kernel launch and print "KERNEL_MS <ms>" to stderr as a stable,
       machine-parseable token (host-side timing only, does not affect the image). */
    cudaEvent_t nb_kern_start, nb_kern_stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&nb_kern_start));
    CUDA_CHECK_RETURN(cudaEventCreate(&nb_kern_stop));
    CUDA_CHECK_RETURN(cudaEventRecord(nb_kern_start));

    nanoBraggSpotsCUDAKernel<<<numBlocks, threadsPerBlock>>>(cu_detector, cu_beam, cu_goniometer, cu_sample, cu_crystal, cu_constants, cu_beam_sources, cu_Fhkl,
            cu_phi_mos_a, cu_phi_mos_b, cu_phi_mos_c, cu_maskimage, cu_floatimage /*out*/, cu_omega_reduction/*out*/, cu_max_I_x_reduction/*out*/, cu_max_I_y_reduction /*out*/, cu_rangemap /*out*/);

    CUDA_CHECK_RETURN(cudaEventRecord(nb_kern_stop));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
    CUDA_CHECK_RETURN(cudaEventSynchronize(nb_kern_stop));
    {
        float nb_kernel_ms = 0.0f;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&nb_kernel_ms, nb_kern_start, nb_kern_stop));
        fprintf(stderr, "KERNEL_MS %f\n", nb_kernel_ms);
    }
    CUDA_CHECK_RETURN(cudaEventDestroy(nb_kern_start));
    CUDA_CHECK_RETURN(cudaEventDestroy(nb_kern_stop));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(floatimage, cu_floatimage, sizeof(*cu_floatimage) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(omega_reduction, cu_omega_reduction, sizeof(*cu_omega_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(max_I_x_reduction, cu_max_I_x_reduction, sizeof(*cu_max_I_x_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(max_I_y_reduction, cu_max_I_y_reduction, sizeof(*cu_max_I_y_reduction) * total_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(rangemap, cu_rangemap, sizeof(*cu_rangemap) * total_pixels, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(cu_detector));
    CUDA_CHECK_RETURN(cudaFree(cu_beam));
    CUDA_CHECK_RETURN(cudaFree(cu_goniometer));
    CUDA_CHECK_RETURN(cudaFree(cu_sample));
    CUDA_CHECK_RETURN(cudaFree(cu_crystal));
    CUDA_CHECK_RETURN(cudaFree(cu_constants));
    CUDA_CHECK_RETURN(cudaFree(cu_beam_sources));
    CUDA_CHECK_RETURN(cudaFree(cu_Fhkl));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_a));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_b));
    CUDA_CHECK_RETURN(cudaFree(cu_phi_mos_c));
    CUDA_CHECK_RETURN(cudaFree(cu_floatimage));
    CUDA_CHECK_RETURN(cudaFree(cu_omega_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_x_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_max_I_y_reduction));
    CUDA_CHECK_RETURN(cudaFree(cu_maskimage));
    CUDA_CHECK_RETURN(cudaFree(cu_rangemap));

    *max_I = 0;
    *max_I_x = 0;
    *max_I_y = 0;
    *sum = 0.0;
    *sumsqr = 0.0;
    *sumn = 0;
    *omega_sum = 0.0;

    for (int i = 0; i < total_pixels; i++) {
        if (!rangemap[i]) {
            continue;
        }
        float pixel = floatimage[i];
        if (pixel > (double) *max_I) {
            *max_I = pixel;
            *max_I_x = max_I_x_reduction[i];
            *max_I_y = max_I_y_reduction[i];
        }
        *sum += pixel;
        *sumsqr += pixel * pixel;
        ++(*sumn);
        *omega_sum += omega_reduction[i];
    }

    delete[] beam_sources;
    delete[] phi_mos_a_host;
    delete[] phi_mos_b_host;
    delete[] phi_mos_c_host;
    free(FhklLinearPad);
    free(rangemap);
    free(omega_reduction);
    free(max_I_x_reduction);
    free(max_I_y_reduction);
}

/* Device code */

/* crystal is const (the kernel never writes it) but deliberately NOT __restrict__: measured on
   sm_120, __restrict__ here costs +1 register for no speed change (it only re-routes ~14 loads
   onto the read-only cache). The register budget takes priority. */
__global__ void nanoBraggSpotsCUDAKernel(const detectorParams * __restrict__ detector, const beamParams * __restrict__ beam, const goniometerParams * __restrict__ goniometer, const sampleParams * __restrict__ sample,
        const crystalParams * crystal, const constParams * __restrict__ constants, const beamSource * __restrict__ beam_sources, const float * __restrict__ Fhkl,
        const float * __restrict__ phi_mos_a, const float * __restrict__ phi_mos_b, const float * __restrict__ phi_mos_c, const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/,
        float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
        float * max_I_y_reduction /*out*/, bool * rangemap) {

    __shared__ float s_Na, s_Nb, s_Nc;

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        s_Na = crystal->Na;
        s_Nb = crystal->Nb;
        s_Nc = crystal->Nc;

    }
    __syncthreads();

    const long total_pixels = detector->spixels * detector->fpixels;
    const long fstride = gridDim.x * blockDim.x;
    const long sstride = gridDim.y * blockDim.y;
    const long stride = fstride * sstride;

    /* add background from something amorphous */
    const float F_bg = sample->water_F;
    const float I_bg = F_bg * F_bg * constants->r_e_sqr * beam->fluence * sample->water_size * sample->water_size * sample->water_size * 1e6f * constants->Avogadro / sample->water_MW;

    for (long pixIdx = (blockDim.y * blockIdx.y + threadIdx.y) * fstride + blockDim.x * blockIdx.x + threadIdx.x; pixIdx < total_pixels; pixIdx += stride) {
        const short fpixel = pixIdx % detector->fpixels;
        const short spixel = pixIdx / detector->fpixels;

        /* allow for just one part of detector to be rendered */
        if (fpixel < detector->roi_xmin || fpixel > detector->roi_xmax || spixel < detector->roi_ymin || spixel > detector->roi_ymax) { /* ROI region of interest */
            continue;
        }

        /* position in pixel array */
        const long j = pixIdx;

        /* allow for the use of a mask */
        if (maskimage != NULL) {
            /* skip any flagged pixels in the mask */
            if (maskimage[j] == 0) {
                continue;
            }
        }

        /* photon accumulator. Water background (I_bg) is not seeded here; it is added
           at the first sub-pixel below, scaled by that pixel's solid angle, so it
           tracks the CPU reference. Folding it into I keeps the kernel register-neutral. */
        float I = 0.0;
        float omega_sub_reduction = 0.0;
        float max_I_x_sub_reduction = 0.0;
        float max_I_y_sub_reduction = 0.0;
        float polar = 1.0;
        bool polar_computed = false;

        /* loop over sub-pixels */
        for (short subS = 0; subS < detector->oversample; ++subS) { /* Y voxel */
            for (short subF = 0; subF < detector->oversample; ++subF) { /* X voxel */
                /* absolute mm position of this sub-pixel on the detector (relative to
                   its origin). Fdet/Sdet are computed in float; this is the start of the
                   pixel_pos -> diffracted -> scattering -> h,k,l -> sincg geometry chain,
                   which is float throughout. */
                const float subpix = detector->subpixel_size;
                float Fdet = subpix * (fpixel * detector->oversample + subF) + subpix / 2.0f; /* X voxel */
                float Sdet = subpix * (spixel * detector->oversample + subS) + subpix / 2.0f; /* Y voxel */

                max_I_x_sub_reduction = Fdet;
                max_I_y_sub_reduction = Sdet;

                for (short thick_tic = 0; thick_tic < detector->detector_thicksteps; ++thick_tic) {
                    /* assume "distance" is to the front of the detector sensor layer */
                    float Odet = (float) thick_tic * detector->detector_thickstep; /* Z Orthogonal voxel. */

                    /* construct detector subpixel position in 3D space: the pix0 origin
                       plus Fdet/Sdet/Odet along the detector basis vectors. */
                    const float p1 = detector->pix0_vector[1], p2 = detector->pix0_vector[2], p3 = detector->pix0_vector[3];
                    float pixel_pos[4];
                    pixel_pos[1] = Fdet * detector->fdet_vector[1] + Sdet * detector->sdet_vector[1] + Odet * detector->odet_vector[1] + p1; /* X */
                    pixel_pos[2] = Fdet * detector->fdet_vector[2] + Sdet * detector->sdet_vector[2] + Odet * detector->odet_vector[2] + p2; /* Y */
                    pixel_pos[3] = Fdet * detector->fdet_vector[3] + Sdet * detector->sdet_vector[3] + Odet * detector->odet_vector[3] + p3; /* Z */
                    pixel_pos[0] = 0.0;

                    if (detector->curved_detector) {
                        /* construct detector pixel that is always "distance" from the sample,
                           operating directly on pixel_pos[]. */
                        float dbvector[4];
                        dbvector[1] = sample->distance * beam->beam_vector[1];
                        dbvector[2] = sample->distance * beam->beam_vector[2];
                        dbvector[3] = sample->distance * beam->beam_vector[3];
                        /* treat detector pixel coordinates as radians */
                        float newvector[4];
                        rotate_axis(dbvector, newvector, detector->sdet_vector, pixel_pos[2] / sample->distance);
                        rotate_axis(newvector, pixel_pos, detector->fdet_vector, pixel_pos[3] / sample->distance);
                    }

                    /* construct the diffracted-beam unit vector to this sub-pixel via
                       unitize(). The same diffracted[] feeds BOTH the correction block
                       (airpath, omega_pixel, parallax, polarization) and the scattering ->
                       h,k,l -> sincg chain. */
                    float diffracted[4];
                    float airpath = unitize(pixel_pos, diffracted);

                    /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                    float omega_pixel = detector->pixel_size * detector->pixel_size / airpath / airpath * sample->close_distance / airpath;
                    /* option to turn off obliquity effect, inverse-square-law only.
                       1.0f (not 1.0): a bare 1.0 is a double literal in C++, and
                       "1.0 / airpath" would silently promote airpath to double for
                       the division -- exactly the double promotion this file must not
                       have. */
                    if (detector->point_pixel) {
                        omega_pixel = 1.0f / airpath / airpath;
                    }

                    /* now calculate detector thickness effects */
                    float capture_fraction = 1.0;
                    if (detector->detector_thick > 0.0f) {
                        /* inverse of effective thickness increase: dot product of odet_vector
                           and diffracted. odet_vector is read via __ldg as a caching hint; it
                           lives in a read-only const-restrict struct, so this changes no results
                           but lets ptxas schedule this cold branch's loads independently of the
                           hot pixel_pos/geometry loads, relieving register pressure. */
                        float parallax = __ldg(&detector->odet_vector[1]) * diffracted[1] + __ldg(&detector->odet_vector[2]) * diffracted[2] + __ldg(&detector->odet_vector[3]) * diffracted[3];
                        capture_fraction = exp(-thick_tic * detector->detector_thickstep * detector->detector_mu / parallax)
                                - exp(-(thick_tic + 1) * detector->detector_thickstep * detector->detector_mu / parallax);
                    }

                    /* Add the water background once, scaled by this sub-pixel's solid
                       angle and capture fraction, so it tracks the CPU reference (which
                       scales the whole pixel intensity by omega, water included). A no-op
                       when there is no water (I_bg == 0). */
                    if (subS == 0 && subF == 0 && thick_tic == 0) {
                        I += capture_fraction * omega_pixel * I_bg;
                    }

                    /* loop over sources now */
                    for (short source = 0; source < beam->sources; ++source) {

                        /* read this source's wavelength via __ldg: beam_sources is constant
                           for the whole launch and every thread reads the same entries, so
                           routing the load through the read-only data cache serves it from
                           cache instead of issuing a regular global load. */
                        float lambda = __ldg(&beam_sources[source].lambda);

                        /* construct the float scattering vector for this pixel from diffracted[].
                           The same scattering[] feeds both the dmin/stol cutoff below and the
                           h,k,l dot product further down. */
                        float scattering[4];
                        scattering[1] = (diffracted[1] - __ldg(&beam_sources[source].neg_unit_source_vector[1])) / lambda;
                        scattering[2] = (diffracted[2] - __ldg(&beam_sources[source].neg_unit_source_vector[2])) / lambda;
                        scattering[3] = (diffracted[3] - __ldg(&beam_sources[source].neg_unit_source_vector[3])) / lambda;

                        /* sin(theta)/lambda is half the scattering vector length */
                        float stol = (float)0.5 * norm3d_fma_rn(scattering[1], scattering[2], scattering[3]);

                        /* rough cut to speed things up when we aren't using whole detector */
                        if (crystal->dmin > 0.0f && stol > 0.0f) {
                            if (crystal->dmin > 0.5f / stol) {
                                continue;
                            }
                        }

                        /* Compute the polarization factor once per pixel (from the first
                           source) and reuse it, matching the CPU reference; the boolean flag
                           keeps the guard uniform across the warp (no divergence). */
                        if (beam->calc_polar && !polar_computed) {
                            /* need to compute polarization factor */
                            float incident[4];
                            incident[1] = __ldg(&beam_sources[source].neg_unit_source_vector[1]);
                            incident[2] = __ldg(&beam_sources[source].neg_unit_source_vector[2]);
                            incident[3] = __ldg(&beam_sources[source].neg_unit_source_vector[3]);
                            polar = polarization_factor(beam->polarization, incident, diffracted, beam->polar_vector);
                            polar_computed = true;
                        }

                        /* sweep over phi angles */
                        for (short phi_tic = 0; phi_tic < goniometer->phisteps; ++phi_tic) {

                            /* enumerate mosaic domains */
                            for (short mos_tic = 0; mos_tic < crystal->mosaic_domains; ++mos_tic) {

                                float h, k, l;
                                {
                                    /* The fully phi+mosaic-rotated cell a/b/c was precomputed on the
                                       host (see the phi_mos_* precompute above) and is looked up here,
                                       so there is no in-kernel rotate_axis or rotate_umat, no sin/cos
                                       and no umat matrix multiply in the hot loop. phi_mos_{a,b,c}
                                       [(phi_tic*mosaic_domains + mos_tic)*3 + {0,1,2}] carry the rotated
                                       a/b/c components {1,2,3}. At phi==0 with no mosaic umat the
                                       rotation is an exact identity, so these equal the base cell
                                       (float)a0/b0/c0 to the bit. */
                                    const int pm = (phi_tic * crystal->mosaic_domains + mos_tic) * 3;
                                    float a[4]; load_rotated_cell_ldg(phi_mos_a, a, pm);
                                    float b[4]; load_rotated_cell_ldg(phi_mos_b, b, pm);
                                    float c[4]; load_rotated_cell_ldg(phi_mos_c, c, pm);

                                    /* construct Miller indices */
                                    h = dot_product(a, scattering);
                                    k = dot_product(b, scattering);
                                    l = dot_product(c, scattering);
                                }

                                /* integer Miller indices of the nearest reflection */
                                short h0 = nearest_hkl(h);
                                short k0 = nearest_hkl(k);
                                short l0 = nearest_hkl(l);

                                /* reduce to fractional Miller indices */
                                float dh = fractional(h);
                                float dk = fractional(k);
                                float dl = fractional(l);

                                /* structure factor of the lattice (parallelepiped crystal)
                                 F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
                                 */
                                float F_latt = 1.0; /* Shape transform for the crystal. */
                                if (crystal->xtal_shape == SQUARE) {
                                    /* xtal is a parallelepiped */
                                    /* delta-reduced sincg: reduce h,k,l with fractional above BEFORE the trig,
                                       then evaluate sinf(PIf*N*delta)/sinf(PIf*delta) */
                                    if (crystal->Na > 1) F_latt *= sincg_delta(dh, (float) crystal->Na);
                                    if (crystal->Nb > 1) F_latt *= sincg_delta(dk, (float) crystal->Nb);
                                    if (crystal->Nc > 1) F_latt *= sincg_delta(dl, (float) crystal->Nc);
                                } else if (crystal->xtal_shape == ROUND) {
                                    /* use sinc3 for elliptical xtal shape,
                                     correcting for sqrt of volume ratio between cube and sphere */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * 0.723601254558268f * sinc3((float) M_PI * sqrt(hrad_sqr * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == GAUSS) {
                                    /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * exp(-(hrad_sqr / 0.63f * 1.0f /*fudge*/));
                                } else if (crystal->xtal_shape == TOPHAT) {
                                    /* make a flat-top spot of same height and volume as square_xtal spots */
                                    float hrad_sqr = (h - h0) * (h - h0) * s_Na * s_Na + (k - k0) * (k - k0) * s_Nb * s_Nb + (l - l0) * (l - l0) * s_Nc * s_Nc;
                                    F_latt = s_Na * s_Nb * s_Nc * (hrad_sqr * 1.0f /*fudge*/ < 0.3969f);
                                }
                                /* structure factor of the unit cell */
                                float F_cell =
                                        crystal->fhklParams.hkls ?
                                                 quickFcell_ldg(crystal->fhklParams.hkls,
                                                    h0,
                                                    crystal->fhklParams.h_max,
                                                    crystal->fhklParams.h_min,
                                                    k0,
                                                    crystal->fhklParams.k_max,
                                                    crystal->fhklParams.k_min,
                                                    l0,
                                                    crystal->fhklParams.l_max,
                                                    crystal->fhklParams.l_min,
                                                    crystal->fhklParams.h_range,
                                                    crystal->fhklParams.k_range,
                                                    crystal->fhklParams.l_range, Fhkl) :
                                                 crystal->default_F;

                                /* now we have the structure factor for this pixel */

                                /* convert amplitudes into intensity (photons per steradian) */
                                I += F_cell * F_cell * F_latt * F_latt * capture_fraction * omega_pixel;
                                omega_sub_reduction += omega_pixel;
                            }
                            /* end of mosaic loop */
                        }
                        /* end of phi loop */
                    }
                    /* end of source loop */
                }
                /* end of detector thickness loop */
            }
            /* end of sub-pixel y loop */
        }
        /* end of sub-pixel x loop */
        /* I holds the Bragg sum plus the solid-angle-scaled water background; apply
           polarization and normalize by the sub-step count. */
        const float photons = (constants->r_e_sqr * beam->fluence * polar * I) / detector->steps;
        floatimage[j] = photons;
        omega_reduction[j] = omega_sub_reduction;
        max_I_x_reduction[j] = max_I_x_sub_reduction;
        max_I_y_reduction[j] = max_I_y_sub_reduction;
        rangemap[j] = true;
    }
}

/* vector inner product where vector magnitude is 0th element */
__device__ __inline__ static float dot_product(const float * x, const float * y) {
    return x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

/* vector cross product where vector magnitude is 0th element */
__device__ static float *cross_product(const float * x, const float * y,
float * z) {
    z[1] = x[2] * y[3] - x[3] * y[2];
    z[2] = x[3] * y[1] - x[1] * y[3];
    z[3] = x[1] * y[2] - x[2] * y[1];
    z[0] = 0.0;

    return z;
}

__device__ __inline__ float norm3d_fma_rn(float v1, float v2, float v3) {
    float q_sqr = v3 * v3;
    q_sqr = __fmaf_rn(v2, v2, q_sqr);
    q_sqr = __fmaf_rn(v1, v1, q_sqr);
    return sqrt(q_sqr);
}

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
__device__ static float unitize(float * vector, float * new_unit_vector) {

    float v1 = vector[1];
    float v2 = vector[2];
    float v3 = vector[3];

    float mag = norm3d_fma_rn(v1, v2, v3);

    if (mag != 0.0f) {
        /* normalize it */
        new_unit_vector[0] = mag;
        new_unit_vector[1] = v1 / mag;
        new_unit_vector[2] = v2 / mag;
        new_unit_vector[3] = v3 / mag;
    } else {
        /* can't normalize, report zero vector */
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;
    }
    return mag;
}

/* rotate a 3-vector about a unit vector axis */
__device__ static float *rotate_axis(const float * __restrict__ v,
float * newv, const float * __restrict__ axis, const float phi) {

    const float sinphi = sin(phi);
    const float cosphi = cos(phi);
    const float a1 = axis[1];
    const float a2 = axis[2];
    const float a3 = axis[3];
    const float v1 = v[1];
    const float v2 = v[2];
    const float v3 = v[3];
    const float dot = (a1 * v1 + a2 * v2 + a3 * v3) * (1.0f - cosphi);

    newv[1] = __fmaf_rn(a1, dot, v1 * cosphi) + __fmaf_rn(-a3, v2, a2 * v3) * sinphi;
    newv[2] = __fmaf_rn(a2, dot, v2 * cosphi) + __fmaf_rn(+a3, v1, -a1 * v3) * sinphi;
    newv[3] = __fmaf_rn(a3, dot, v3 * cosphi) + __fmaf_rn(-a2, v1, a1 * v2) * sinphi;

    return newv;
}

__device__ __inline__ static long flatten3dindex(short x, short y, short z, short x_range, short y_range, short z_range) {
    return x * y_range * z_range + y * z_range + z;
}

__device__ __inline__ float quickFcell_ldg(short hkls, short h0, short h_max, short h_min, short k0, short k_max, short k_min, short l0, short l_max,
        short l_min, short h_range,
        short k_range, short l_range, const float * __restrict__ Fhkl) {
    short h = min(max(h0 - h_min, 0), h_range - 1);
    short k = min(max(k0 - k_min, 0), k_range - 1);
    short l = min(max(l0 - l_min, 0), l_range - 1);
    return __ldg(&Fhkl[flatten3dindex(h, k, l, h_range, k_range, l_range)]);
}

/* Copy one rotated cell vector (three components) from the host-precomputed table into
   out[]. Vectors in this codebase use a 4-element convention: element 0 holds the
   magnitude and elements 1..3 hold the x,y,z components. The magnitude is not needed
   here, so out[0] is set to 0 and out[1], out[2], out[3] receive the three components,
   read through the __ldg read-only cache. idx_base selects the table entry:
   (phi_tic * mosaic_domains + mos_tic) * 3. */
__device__ __forceinline__ void load_rotated_cell_ldg(const float * __restrict__ tbl, float out[4], int idx_base) {
    out[0] = 0.0f;
    out[1] = __ldg(&tbl[idx_base + 0]);
    out[2] = __ldg(&tbl[idx_base + 1]);
    out[3] = __ldg(&tbl[idx_base + 2]);
}

/* nearest_hkl: integer Miller index of the nearest reciprocal-lattice point
   (the Fhkl structure-factor lookup index) */
__device__ __forceinline__ static short nearest_hkl(float h) {
    return (short) ceilf(h - 0.5f);
}

/* fractional: distance from the nearest Bragg peak, reduced to |delta| <= 0.5 */
__device__ __forceinline__ static float fractional(float h) {
    return h - rintf(h);
}

/* Delta-reduced sincg: evaluates the N-slit interference function
   sin(pi*N*delta)/sin(pi*delta) from an already-reduced argument
   delta = h - rint(h), so |delta| <= 0.5. Reducing before the trig calls keeps
   the argument small and well-conditioned. F_latt is squared into intensity
   downstream, so the sign difference vs evaluating at the unreduced argument
   is irrelevant. */
__device__ __inline__ static float sincg_delta(float delta, float N) {
    /* removable singularity: as delta->0, sin(pi*N*delta)/sin(pi*delta) -> N.
       Guard delta==0 (Bragg peak / integer h) to avoid 0/0 = NaN. */
    const float PIf = 3.14159265358979323846f;
    if (delta == 0.0f)
        return N;
    return sinf(PIf * N * delta) / sinf(PIf * delta);
}

/* Fourier transform of a sphere */
__device__ static float sinc3(float x) {
    if (x != 0.0f)
        return 3.0f * (sin(x) / x - cos(x)) / (x * x);

    return 1.0;

}

/* polarization factor */
__device__ static float polarization_factor(float kahn_factor, const float * __restrict__ unitIncident, float *unitDiffracted,
        const float * __restrict__ unitAxis) {
    float cos2theta, cos2theta_sqr, sin2theta_sqr;
    float psi = 0.0f;
    float E_in[4], B_in[4], E_out[4], B_out[4];

    /* component of diffracted unit vector along incident beam unit vector */
    cos2theta = dot_product(unitIncident, unitDiffracted);
    cos2theta_sqr = cos2theta * cos2theta;
    sin2theta_sqr = 1.0f - cos2theta_sqr;

    if (kahn_factor != 0.0f) {
        /* tricky bit here is deciding which direction the E-vector lies in for each source
         here we assume it is closest to the "axis" defined above */

        /* cross product to get "vertical" axis that is orthogonal to the canonical "polarization" */
        cross_product(unitAxis, unitIncident, B_in);
        /* make it a unit vector */
        unitize(B_in, B_in);

        /* cross product with incident beam to get E-vector direction */
        cross_product(unitIncident, B_in, E_in);
        /* make it a unit vector */
        unitize(E_in, E_in);

        /* get components of diffracted ray projected onto the E-B plane */
        E_out[0] = dot_product(unitDiffracted, E_in);
        B_out[0] = dot_product(unitDiffracted, B_in);

        /* compute the angle of the diffracted ray projected onto the incident E-B plane */
        psi = -atan2(B_out[0], E_out[0]);
    }

    /* correction for polarized incident beam */
    return 0.5f * (1.0f + cos2theta_sqr - kahn_factor * cos(2.0f * psi) * sin2theta_sqr);
}
