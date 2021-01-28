#include <iostream>
#include <vector>
using namespace  std;
// 724. 寻找数组的中心索引 easy
// 2021年1月28日23点29分
int main() {
    vector<int> nums={1,7,3,6,5,6};
    if(nums.empty())
        return -1;
    int all=nums[0];
    vector<int> a;
    a.push_back(0);
    for(int i=1;i<nums.size();i++){
        a.push_back(a.back()+nums[i-1]);
        all+=nums[i];
    }
    for(int i=0;i<nums.size();i++){
        if(all-nums[i]-a[i]==a[i]){
            return i;
        }
    }
    return -1;
}
