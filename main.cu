///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	ENTREGABLE 1 : Temporización GPU
/// Autor:		Angel Sierra Gomez, Gustavo Gutierrez Martin
/// Fecha:		Noviembre 2022
///////////////////////////////////////////////////////////////////////////

/// dependencias ///
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

/// constantes ///
#define MB (1<<20) /// MiB = 2^20
#define ROWS 7
#define COLUMNS 24

/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);
/// realiza la suma de los arrays en el device
__global__ void transfer(const int *dev_vector, int *dev_result);

int main() {
    int deviceCount;
    int *hst_vector,*hst_result;
    int *dev_vector,*dev_result;
    dim3 blocks(1);
    dim3 threads(ROWS, COLUMNS);
    /// declaracion de eventos
    cudaEvent_t start;
    cudaEvent_t stop;

    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        printf("¡No se ha encontrado un dispositivo CUDA!\n");
        printf("<pulsa [INTRO] para finalizar>");
        getchar();
        return 1;
    } else {
        printf("Se han encontrado %d dispositivos CUDA:\n", deviceCount);
        for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
            ///obtenemos las propiedades del dispositivo CUDA
            cudaDeviceProp deviceProp{};
            cudaGetDeviceProperties(&deviceProp, deviceID);
            int SM = deviceProp.multiProcessorCount;
            int cudaCores = getCudaCores(deviceProp);
            printf("***************************************************\n");
            printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
            printf("***************************************************\n");
            printf("- Capacidad de Computo            \t: %d.%d\n", deviceProp.major, deviceProp.minor);
            printf("- No. de MultiProcesadores        \t: %d \n", SM);
            printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
            printf("- Memoria Global (total)          \t: %zu MiB\n", deviceProp.totalGlobalMem / MB);
            printf("- No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
            printf("***************************************************\n");
        }
    }
    /// reserva del espacio de memoria en el host
    hst_vector = (int*)malloc( ROWS * COLUMNS * sizeof(int));
    hst_result = (int*)malloc( ROWS * COLUMNS * sizeof(int));
    /// reserva del espacio de memoria en el device
    cudaMalloc( (void**)&dev_vector, ROWS * COLUMNS * sizeof(int) );
    cudaMalloc( (void**)&dev_result, ROWS * COLUMNS * sizeof(int) );
    /// cargamos los datos iniciales en el host
    srand ( (int)time(nullptr) );
    for (int i = 0; i < ROWS; i++)  {
        int number = (int) rand() % 9 + 1;
        for (int j=0; j < COLUMNS; j++) {
            /// inicializamos hst_vector1 con numeros aleatorios entre 0 y 1
            hst_vector[i * COLUMNS + j] = number;
            /// inicializamos hst_vector2 con ceros
            hst_result[i * COLUMNS + j] = 0;
        }
    }
    /// creacion de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /// transfiere datos de host a device
    cudaMemcpy(dev_vector,hst_vector, ROWS * COLUMNS * sizeof(int),cudaMemcpyHostToDevice);
    /// mostramos los datos con los que llamamos al device
    printf("Lanzamiento de: %d bloque y %d hilos \n", 1, threads.x * threads.y);
    printf("> Eje X: %d \n", threads.x);
    printf("> Eje Y: %d \n", threads.y);
    printf("***************************************************\n");
    /// marca de inicio
    cudaEventRecord(start,nullptr);
    /// sumamos los items
    transfer<<< blocks, threads >>>(dev_vector, dev_result);
    /// marca de final
    cudaEventRecord(stop,nullptr);
    /// sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    /// cálculo del tiempo en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    /// impresion de resultados
    printf("> Tiempo de ejecucion: %f ms\n",elapsedTime);
    printf("***************************************************\n");
    /// transferimos los datos del device al host
    cudaMemcpy(hst_result, dev_result, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);
    /// muestra por pantalla los datos del host
    printf("MATRIZ ORIGINAL:\n");
    for (int i = 0; i < ROWS; i++)  {
        for (int j = 0; j < COLUMNS; j++) {
            printf("%d ", hst_vector[j + i * COLUMNS]);
        }
        printf("\n");
    }
    printf("\n");
    printf("MATRIZ FINAL:\n");
    for (int i = 0; i < ROWS; i++)  {
        for (int j = 0; j < COLUMNS; j++) {
            printf("%d ", hst_result[j + i * COLUMNS]);
        }
        printf("\n");
    }
    printf("\n");

    /// función que muestra por pantalla la salida del programa
    time_t fecha;
    time(&fecha);
    printf("***************************************************\n");
    printf("Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    printf("<pulsa [INTRO] para finalizar>");
    getchar();

    /// liberacion de recursos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_vector);
    cudaFree(dev_result);
    return 0;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

__global__ void transfer(const int *dev_vector, int *dev_result) {
    /// identificador del hilo
    unsigned int threadX = threadIdx.y;
    unsigned int threadY = threadIdx.x;
    /// calculamos el ID  hilo
    unsigned int myID = threadY + threadX * blockDim.x;
    /// calculamos la fila donde se encuentra la posicion
    int row = (int) myID / COLUMNS;
    /// calculamos si la posicion
    if (row < (ROWS - 1)) {
        dev_result[myID + COLUMNS] = dev_vector[myID];
    } else {
        dev_result[myID - (COLUMNS * (ROWS - 1))] = dev_vector[myID];
    }
}
