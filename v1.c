#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

    // Ajustar tamaños para aligned_alloc (múltiplos de ALIGN)
    size_t sizeA = n * n * sizeof(double);
    size_t sizeB = n * sizeof(double);

    sizeA = ((sizeA + ALIGN - 1) / ALIGN) * ALIGN;
    sizeB = ((sizeB + ALIGN - 1) / ALIGN) * ALIGN;

    double *A = aligned_alloc(ALIGN, sizeA);
    double *b = aligned_alloc(ALIGN, sizeB);
    double *x = aligned_alloc(ALIGN, sizeB);
    double *xNew = aligned_alloc(ALIGN, sizeB);

    if (!A || !b || !x || !xNew) {
        printf("Error en la asignación de memoria\n");
        return 1;
    }

    // Inicialización de A (diagonal dominante)
    for (int i = 0; i < n; i++) {

        double sum = 0.0;
        double *Ai = &A[i * n];  // mejora de acceso (no cambia algoritmo)

        for (int j = 0; j < n; j++) {

            double val = (double)rand() / RAND_MAX;

            Ai[j] = val;

            if (i != j)
                sum += fabs(val);
        }

        Ai[i] = sum + 1.0;
    }

    // Inicialización de b
    for (int i = 0; i < n; i++) {
        b[i] = (double)rand() / RAND_MAX;
    }

    // Inicialización de x y xNew
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        xNew[i] = 0.0;
    }

    start_counter();

    int iter;

    for (iter = 0; iter < MAX_ITER; iter++) {

        norma = 0.0;

        for (int i = 0; i < n; i++) {

            double sigma = 0.0;
            double *Ai = &A[i * n];  // acceso más eficiente

            for (int j = 0; j < n; j++) {

                if (i != j)
                    sigma += Ai[j] * x[j];
            }

            xNew[i] = (b[i] - sigma) / Ai[i];

            double diff = xNew[i] - x[i];
            norma += diff * diff;
        }

        // Copia xNew -> x
        for (int i = 0; i < n; i++)
            x[i] = xNew[i];

        // Evitar sqrt en cada iteración (más eficiente)
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