//
//  sorting.cpp
//  12.16
//
//  Created by 蓝蒋 on 2023/6/23.
//  实验经典排序算法

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 100 // 定义待排序数组的长度
#define RANDOM_SEED 12345
#define MIN_VALUE -100
#define MAX_VALUE 100
int generateRandomNumber(int min, int max) {
    return min + rand() % (max - min + 1);
}
// 冒泡排序
void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // 交换元素
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
void selectionSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        int minIndex = i;

        // 找到未排序部分中的最小元素的索引
        for (int j = i + 1; j < size; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        // 将最小元素与当前位置交换
        int temp = arr[i];
        arr[i] = arr[minIndex];
        arr[minIndex] = temp;
    }
}
void insertionSort(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        int key = arr[i];
        int j = i - 1;
        
        // 将比 key 大的元素向后移动
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        
        arr[j + 1] = key;
    }
}
void shellSort(int arr[], int size) {
    // 定义间隔序列，可以根据需要进行调整
    int gap = size / 2;
    
    while (gap > 0) {
        for (int i = gap; i < size; i++) {
            int temp = arr[i];
            int j = i;
            
            // 在当前间隔下进行插入排序
            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            
            arr[j] = temp;
        }
        
        // 缩小间隔
        gap /= 2;
    }
}
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];  // 选择最后一个元素作为基准
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);

        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
    }
}

void quickSort2(int arr[], int size) {
    quickSort(arr, 0, size - 1);
}
// 计算排序过程中的存储空间大小
int calculateSpaceComplexity(int size) {
    // 这里假设每个数组元素占用 4 个字节的空间
    int dataSize = size * sizeof(int);
    int tempSize = size * sizeof(int); // 用于临时存储元素的空间大小
    int totalSize = dataSize + tempSize; // 总的存储空间大小
    return totalSize;
}

int main(void) {
    int arr[ARRAY_SIZE]; // 待排序的数组

    
    //改为固定随机数种子
    srand(RANDOM_SEED);
    // 生成随机数并放入数组
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = generateRandomNumber(MIN_VALUE, MAX_VALUE);
    }

    // 打印待排序数据
    printf("待排序数组：\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // 计算排序过程中的存储空间大小
    int spaceComplexity = calculateSpaceComplexity(400);
    printf("排序过程中的存储空间大小：%d 字节\n", spaceComplexity);

    // 排序并记录执行时间
    clock_t startTime = clock();
    quickSort2(arr, ARRAY_SIZE);
    clock_t endTime = clock();

    // 打印排序结果和执行时间
    printf("排序后的数组：\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // 计算执行时间（毫秒）
    double duration = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    printf("执行时间（毫秒）：%lf\n", duration);

    return 0;
}
