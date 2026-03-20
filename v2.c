#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "counter.h"

#define TOL 1e-5
#define MAX_ITER 15000
#define ALIGN 32

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Uso: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 1) {
        printf("Introduce un tamaño válido.\n");
        return 1;
    }

    srand(0);

    double norma = 0.0;
    double ck;

    // Ajuste tamaños aligned_alloc
    size_t sizeA = n * n * sizeof(double);
    size_t sizeV = n * sizeof(double);

    sizeA = ((sizeA + ALIGN - 1) / ALIGN) * ALIGN;
    sizeV = ((sizeV + ALIGN - 1) / ALIGN) * ALIGN;

    double *A = aligned_alloc(ALIGN, sizeA);
    double *b = aligned_alloc(ALIGN, sizeV);
    double *x = aligned_alloc(ALIGN, sizeV);
    double *xNew = aligned_alloc(ALIGN, sizeV);

    if (!A || !b || !x || !xNew) {
        printf("Error en memoria\n");
        return 1;
    }

    // Inicialización matriz diagonal dominante
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        double *Ai = &A[i * n];

        for (int j = 0; j < n; j++) {
            double val = (double)rand() / RAND_MAX;
            Ai[j] = val;

            if (i != j)
                sum += fabs(val);
        }

        Ai[i] = sum + 1.0;
    }

    // Inicializar b
    for (int i = 0; i < n; i++)
        b[i] = (double)rand() / RAND_MAX;

    // Inicializar x
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        xNew[i] = 0.0;
    }

    start_counter();

    int iter;

    for (iter = 0; iter < MAX_ITER; iter++) {

        norma = 0.0;

        for (int i = 0; i < n; i++) {

            double *Ai = &A[i * n];
            double sigma = 0.0;

            int j;

            // Parte 1: j < i (sin if)
            for (j = 0; j <= i - 4; j += 4) {
                sigma += Ai[j] * x[j];
                sigma += Ai[j + 1] * x[j + 1];
                sigma += Ai[j + 2] * x[j + 2];
                sigma += Ai[j + 3] * x[j + 3];
            }
            for (; j < i; j++) {
                sigma += Ai[j] * x[j];
            }

            // Parte 2: j > i (sin if)
            for (j = i + 1; j <= n - 4; j += 4) {
                sigma += Ai[j] * x[j];
                sigma += Ai[j + 1] * x[j + 1];
                sigma += Ai[j + 2] * x[j + 2];
                sigma += Ai[j + 3] * x[j + 3];
            }
            for (; j < n; j++) {
                sigma += Ai[j] * x[j];
            }

            double xi_new = (b[i] - sigma) / Ai[i];
            double diff = xi_new - x[i];

            xNew[i] = xi_new;
            norma += diff * diff;
        }

        // Intercambio de punteros (evita copia)
        double *tmp = x;
        x = xNew;
        xNew = tmp;

        if (norma < TOL * TOL)
            break;
    }

    ck = get_counter();

    printf("Convergió en %d iteraciones con error %e\n", iter, sqrt(norma));
    printf("Ciclos: %.2lf\n", ck);

    free(A);
    free(b);
    free(x);
    free(xNew);

    return 0;
}
