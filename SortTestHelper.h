//
// Created by Administrator on 2021/5/13.
//

#ifndef UNTITLED_SORTTESTHELPER_H
#define UNTITLED_SORTTESTHELPER_H

#include <iostream>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <cassert>

using namespace std;
namespace SortTestHelper {
    template<typename T>
    vector<T> generateRandomArray(int n, int rangeL, int rangeR) {
        assert(rangeL <= rangeR);
        vector<T> a(n);
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            a[i] = rand() % (rangeR - rangeL + 1) + rangeL;
        }
        return a;
    }

    template<typename T>
    void printArray(vector<T> a) {
        for (int i = 0; i < a.size(); i++) {
            cout << a[i] << " ";
        }
    }

    template<typename T>
    bool isSorted(vector<T> arr) {
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i - 1] > arr[i])
                return false;
        }
        return true;
    }

    template<typename T>
    void testSort(string sortName, void(*sort)(vector<T> &), vector<T> arr) {
        clock_t startTime = clock();
        sort(arr);
        clock_t stopTime = clock();
        assert(isSorted(arr));
        cout << sortName << ": " << double(stopTime - startTime) / CLOCKS_PER_SEC << " s" << endl;
        return;
    }

    template<typename T>
    vector<T> generateNearlyOrderedArray(int n, int swapTimes) {
        vector<T> a;
        for (int i = 0; i < n; i++) {
            a.push_back(i);
        }
        srand(time(NULL));
        for (int i = 0; i < swapTimes; i++) {
            int posx = rand() % n;
            int posy = rand() % n;
            swap(a[posx], a[posy]);
        }
        return a;
    }
}

#endif //UNTITLED_SORTTESTHELPER_H
